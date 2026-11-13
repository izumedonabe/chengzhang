//! Byte-level BPE tokenizer for the Qwen family, without the `tokenizers` crate (MBs).
//! Loads either `tokenizer.json` (Qwen3.5) or `vocab.json` + `merges.txt` +
//! `tokenizer_config.json` (Qwen3-ASR). The pre-tokenizer regex is reproduced by a
//! hand-written scanner (see `split`), which matches the regex's alternation order.
//! Not reproduced: the NFC normalizer (dictation text is already NFC in practice).
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use unicode_general_category::{get_general_category as gc, GeneralCategory as G};

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    id_to_token: Vec<String>,
    specials: Vec<(String, u32)>, // sorted longest first
    special_ids: HashMap<u32, String>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    marks_are_letters: bool, // Qwen3.5 pattern uses [\p{L}\p{M}]+, Qwen2 uses \p{L}+
    cache: std::cell::RefCell<HashMap<String, Vec<u32>>>,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap();
    }
    out
}

fn read_json(p: &Path) -> Result<Value, String> {
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&t).map_err(|e| format!("{}: {e}", p.display()))
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Tokenizer, String> {
        let (vocab, merges, specials, pattern): (HashMap<String, u32>, Vec<(String, String)>, Vec<(String, u32)>, String);
        if dir.join("tokenizer.json").exists() {
            let t = read_json(&dir.join("tokenizer.json"))?;
            let m = &t["model"];
            vocab = serde_json::from_value(m["vocab"].clone()).map_err(|e| e.to_string())?;
            merges = m["merges"].as_array().ok_or("no merges")?.iter().filter_map(|x| match x {
                Value::Array(a) => Some((a[0].as_str()?.to_string(), a[1].as_str()?.to_string())),
                Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())),
                _ => None,
            }).collect();
            specials = t["added_tokens"].as_array().map(|a| a.iter().filter_map(|x| {
                Some((x["content"].as_str()?.to_string(), x["id"].as_u64()? as u32))
            }).collect()).unwrap_or_default();
            pattern = t["pre_tokenizer"].to_string();
        } else {
            vocab = serde_json::from_value(read_json(&dir.join("vocab.json"))?).map_err(|e| e.to_string())?;
            let mt = std::fs::read_to_string(dir.join("merges.txt")).map_err(|e| e.to_string())?;
            merges = mt.lines().filter(|l| !l.starts_with("#version") && !l.is_empty())
                .filter_map(|l| l.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))).collect();
            let cfg = read_json(&dir.join("tokenizer_config.json"))?;
            specials = cfg["added_tokens_decoder"].as_object().map(|o| o.iter().filter_map(|(id, v)| {
                Some((v["content"].as_str()?.to_string(), id.parse().ok()?))
            }).collect()).unwrap_or_default();
            pattern = String::new();
        }
        let n = vocab.values().copied().chain(specials.iter().map(|s| s.1)).max().unwrap_or(0) as usize + 1;
        let mut id_to_token = vec![String::new(); n];
        for (t, &i) in &vocab {
            id_to_token[i as usize] = t.clone();
        }
        let mut specials = specials;
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let special_ids = specials.iter().map(|(s, i)| (*i, s.clone())).collect();
        let byte_enc = bytes_to_unicode();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let ranks = merges.into_iter().enumerate().map(|(r, m)| (m, r as u32)).collect();
        Ok(Tokenizer {
            vocab, ranks, id_to_token, specials, special_ids, byte_enc, byte_dec,
            marks_are_letters: pattern.contains("\\\\p{M}"),
            cache: Default::default(),
        })
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.specials.iter().find(|x| x.0 == s).map(|x| x.1).or_else(|| self.vocab.get(s).copied())
    }

    /// Encode text. With `special`, special-token strings ("<|im_start|>"…) become their ids.
    pub fn encode(&self, text: &str, special: bool) -> Vec<u32> {
        let mut out = vec![];
        let mut rest = text;
        while !rest.is_empty() {
            let next = if special {
                self.specials.iter().filter_map(|(s, id)| rest.find(s.as_str()).map(|p| (p, s.len(), *id)))
                    .min_by_key(|(p, len, _)| (*p, usize::MAX - len))
            } else {
                None
            };
            match next {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in split(text, self.marks_are_letters) {
            if let Some(ids) = self.cache.borrow().get(piece) {
                out.extend_from_slice(ids);
                continue;
            }
            let ids = self.bpe(piece);
            out.extend_from_slice(&ids);
            if self.cache.borrow().len() < 50_000 {
                self.cache.borrow_mut().insert(piece.to_string(), ids);
            }
        }
    }

    fn bpe(&self, piece: &str) -> Vec<u32> {
        let mut parts: Vec<String> = piece.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map(|b| r < b.0).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }
        parts.iter().filter_map(|p| self.vocab.get(p).copied()).collect()
    }

    /// Raw bytes of one token (may be a partial UTF-8 sequence).
    pub fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        if let Some(s) = self.special_ids.get(&id) {
            out.extend_from_slice(s.as_bytes());
            return;
        }
        if let Some(t) = self.id_to_token.get(id as usize) {
            out.extend(t.chars().filter_map(|c| self.byte_dec.get(&c)));
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = vec![];
        for &i in ids {
            self.token_bytes(i, &mut b);
        }
        String::from_utf8_lossy(&b).into_owned()
    }
}

// ----------------------------------------------------------------------------- pre-tokenizer

fn is_letter(c: char) -> bool {
    matches!(gc(c), G::UppercaseLetter | G::LowercaseLetter | G::TitlecaseLetter | G::ModifierLetter | G::OtherLetter)
}
fn is_mark(c: char) -> bool {
    matches!(gc(c), G::NonspacingMark | G::SpacingMark | G::EnclosingMark)
}
fn is_number(c: char) -> bool {
    matches!(gc(c), G::DecimalNumber | G::LetterNumber | G::OtherNumber)
}

/// Split like the Qwen pre-tokenizer regex (alternatives tried in order at each position):
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
///   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
/// (Qwen2/Qwen3-ASR: the same without \p{M}.)
pub fn split(text: &str, marks: bool) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { Some(cs[i].1) } else { None };
    let pos = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let letterish = |c: char| is_letter(c) || (marks && is_mark(c));
    let other = |c: char| !c.is_whitespace() && !is_letter(c) && !is_number(c) && !(marks && is_mark(c));
    let mut out = vec![];
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let start = i;
        let mut end = None;
        // 1. contractions
        if c == '\'' {
            let l1 = at(i + 1).map(|x| x.to_ascii_lowercase());
            let l2 = at(i + 2).map(|x| x.to_ascii_lowercase());
            if matches!(l1, Some('s' | 't' | 'm' | 'd')) {
                end = Some(i + 2);
            }
            if matches!((l1, l2), (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l'))) {
                end = Some(i + 3);
            }
        }
        // 2. optional prefix + letters
        if end.is_none() {
            let mut j = i;
            if !letterish(c) && c != '\r' && c != '\n' && !is_letter(c) && !is_number(c) && at(i + 1).is_some_and(letterish) {
                j = i + 1;
            }
            if at(j).is_some_and(letterish) {
                while at(j).is_some_and(letterish) {
                    j += 1;
                }
                end = Some(j);
            }
        }
        // 3. a single number char
        if end.is_none() && is_number(c) {
            end = Some(i + 1);
        }
        // 4. optional space + punctuation run + newlines
        if end.is_none() {
            let mut j = i;
            if c == ' ' && at(i + 1).is_some_and(other) {
                j = i + 1;
            }
            if at(j).is_some_and(other) {
                while at(j).is_some_and(other) {
                    j += 1;
                }
                while matches!(at(j), Some('\r' | '\n')) {
                    j += 1;
                }
                end = Some(j);
            }
        }
        // 5–7. whitespace
        if end.is_none() && c.is_whitespace() {
            let mut j = i;
            while at(j).is_some_and(|x| x.is_whitespace()) {
                j += 1;
            }
            if let Some(k) = (i..j).rev().find(|&k| matches!(cs[k].1, '\r' | '\n')) {
                end = Some(k + 1); // \s*[\r\n]+
            } else if j == n || j - i == 1 {
                end = Some(j); // \s+(?!\S) at the end, or a single space: \s+
            } else {
                end = Some(j - 1); // leave the last space to attach to the next word
            }
        }
        let e = end.unwrap_or(i + 1);
        out.push(&text[pos(start)..pos(e)]);
        i = e;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn splits_like_the_regex() {
        assert_eq!(split("Hello world  foo", true), vec!["Hello", " world", " ", " foo"]);
        assert_eq!(split("我们2026年的目标", true), vec!["我们", "2", "0", "2", "6", "年的目标"]);
        assert_eq!(split("don't stop!!\n\nok", true), vec!["don", "'t", " stop", "!!\n\n", "ok"]);
        assert_eq!(split("a\n  b", true), vec!["a", "\n", " ", " b"]);
        assert_eq!(split("PR，你", true), vec!["PR", "，你"]);
    }
}
