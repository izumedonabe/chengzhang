//! Learning from the user's corrections. After a dictation is pasted the user fixes it in
//! place and presses the correction hotkey. We diff what was pasted against what is there
//! now and keep the replaced pieces ("read is" → "Redis", "成长" → "成章"):
//!   - the pairs go into the polishing prompt, and pairs with English in them are also
//!     replaced directly in later transcripts;
//!   - the corrected spelling of terms goes into the personal dictionary, which biases
//!     Qwen3-ASR itself.

use crate::text::norm;
use std::collections::HashSet;

/// (what was recognized, what the user wanted)
pub type Pair = (String, String);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Learned {
    /// The part of the edited text that corresponds to the dictation.
    pub corrected: String,
    /// Anything changed at all (punctuation included).
    pub changed: bool,
    pub pairs: Vec<Pair>,
    /// Dictionary candidates: the corrected spelling of names and terms.
    pub terms: Vec<String>,
}

/// Longest side of a pair we keep; longer edits are rewrites, not recognition errors.
const MAX_SIDE: usize = 24;
/// Bound on the diff table (chars × chars).
const MAX_CELLS: usize = 6_000_000;

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F)
}

fn word(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn has_ascii_word(s: &str) -> bool {
    s.chars().any(word)
}

/// The part of `b` that can hold `a`: all of it when short, else the stretch sharing the
/// most character bigrams with `a`, plus a margin. Keeps a whole document from being
/// matched char by char against a short dictation.
fn window(a: &[char], b: &[char]) -> (usize, usize) {
    if b.len() <= a.len() * 2 + 40 || a.len() < 2 {
        return (0, b.len());
    }
    let grams: HashSet<(char, char)> = a.windows(2).map(|w| (w[0], w[1])).collect();
    let hits: Vec<u32> = b.windows(2).map(|w| grams.contains(&(w[0], w[1])) as u32).collect();
    let w = a.len().min(hits.len()).max(1);
    let mut sum: u32 = hits[..w].iter().sum();
    let (mut best, mut at) = (sum, 0);
    for i in w..hits.len() {
        sum = sum + hits[i] - hits[i - w];
        if sum > best {
            best = sum;
            at = i + 1 - w;
        }
    }
    let margin = a.len() / 2 + 20;
    (at.saturating_sub(margin), (at + w + margin).min(b.len()))
}

/// Matched index pairs of a longest common subsequence. None when too large.
fn lcs(a: &[char], b: &[char]) -> Option<Vec<(usize, usize)>> {
    let (n, m) = (a.len(), b.len());
    if n.saturating_mul(m) > MAX_CELLS {
        return None;
    }
    let w = m + 1;
    let mut dp = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * w + j] = if a[i] == b[j] {
                dp[(i + 1) * w + j + 1] + 1
            } else {
                dp[(i + 1) * w + j].max(dp[i * w + j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, vec![]);
    while i < n && j < m {
        if a[i] == b[j] && dp[i * w + j] == dp[(i + 1) * w + j + 1] + 1 {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if dp[(i + 1) * w + j] >= dp[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(out)
}

/// How well `orig` (letters and digits) lines up with `edited`, 0..1: the share of the
/// shorter of the two found in order in the other. `edited` can be the dictation inside a
/// longer text box, or just the part of it the user selected. Used to find which
/// dictation an edited text belongs to.
pub fn similarity(orig: &str, edited: &str) -> f32 {
    let a: Vec<char> = norm(orig).chars().collect();
    let b: Vec<char> = norm(edited).chars().collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (s, e) = window(&a, &b);
    let shorter = a.len().min(e - s);
    lcs(&a, &b[s..e]).map(|m| m.len() as f32 / shorter.max(1) as f32).unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug)]
struct Edit {
    a0: usize,
    a1: usize,
    b0: usize,
    b1: usize,
    /// widened with a neighbouring character for context (a one-character Chinese fix)
    grown: bool,
}

const PUNCT: &str = "，。、；：？！“”‘’（）《》【】「」…—,;:?!\"'()[]<>";

fn trim_side(s: &str) -> String {
    s.trim_start_matches(|c: char| c.is_whitespace() || PUNCT.contains(c))
        .trim_end_matches(|c: char| c.is_whitespace() || c == '.' || PUNCT.contains(c))
        .to_string()
}

/// Letters and digits, case kept.
fn core(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// Diff `orig` (what was pasted) against `edited` (what is in the text box now, possibly
/// with text around it, or only the part the user selected). None when the two don't
/// line up at all or are too long to compare.
pub fn learn(orig: &str, edited: &str) -> Option<Learned> {
    let a: Vec<char> = orig.chars().collect();
    let all_b: Vec<char> = edited.chars().collect();
    let (ws, we) = window(&a, &all_b);
    let b = &all_b[ws..we];
    let (n, m) = (a.len(), b.len());
    let matched = lcs(&a, b)?;
    if matched.is_empty() {
        return None;
    }

    // gaps between runs of matched characters are the edits
    let mut edits: Vec<Edit> = vec![];
    let (mut pa, mut pb) = (0, 0);
    for &(i, j) in matched.iter().chain(std::iter::once(&(n, m))) {
        if i > pa || j > pb {
            edits.push(Edit { a0: pa, a1: i, b0: pb, b1: j, grown: false });
        }
        pa = i + 1;
        pb = j + 1;
    }

    // text before/after the dictation in the box (pure insertion at an edge), or a
    // selection that covers only part of it (pure deletion at an edge): not edits
    let (mut cs, mut ce) = (0, m);
    if let Some(e) = edits.first().copied() {
        if e.a0 == 0 && e.b0 == 0 && (e.a0 == e.a1 || e.b0 == e.b1) {
            cs = e.b1;
            edits.remove(0);
        }
    }
    if let Some(e) = edits.last().copied() {
        if e.a1 == n && e.b1 == m && (e.a0 == e.a1 || e.b0 == e.b1) {
            ce = e.b0;
            edits.pop();
        }
    }
    let corrected: String = b[cs..ce.max(cs)].iter().collect();
    let changed = !edits.is_empty();

    // join edits split by a single shared character, or by the shared letters of one
    // English word ("post gray sql" → "PostgreSQL" diffs as p→P, " "→"", "ay sql"→"eSQL")
    let merge = |edits: Vec<Edit>, gap_ok: &dyn Fn(&Edit, &Edit) -> bool| {
        let mut out: Vec<Edit> = vec![];
        for e in edits {
            match out.last_mut() {
                Some(p) if e.a0 <= p.a1 || gap_ok(p, &e) => {
                    p.a1 = p.a1.max(e.a1);
                    p.b1 = p.b1.max(e.b1);
                    p.grown |= e.grown;
                }
                _ => out.push(e),
            }
        }
        out
    };
    let mut edits = merge(edits, &|p, e| {
        let gap = &a[p.a1..e.a0];
        gap.len() <= 1 || (gap.len() <= 12 && gap.iter().all(|&c| word(c) || c == ' '))
    });

    for e in edits.iter_mut() {
        let touches_word = |e: &Edit, left: bool| {
            let (ca, cb) = if left {
                (a.get(e.a0).filter(|_| e.a0 < e.a1), b.get(e.b0).filter(|_| e.b0 < e.b1))
            } else {
                (e.a1.checked_sub(1).filter(|_| e.a1 > e.a0).map(|i| &a[i]),
                 e.b1.checked_sub(1).filter(|_| e.b1 > e.b0).map(|i| &b[i]))
            };
            ca.is_some_and(|&c| word(c)) || cb.is_some_and(|&c| word(c))
        };
        // English: widen to whole words ("r"→"R" in "rust" becomes "rust"→"Rust")
        while e.a0 > 0 && e.b0 > 0 && a[e.a0 - 1] == b[e.b0 - 1] && word(a[e.a0 - 1]) && touches_word(e, true) {
            e.a0 -= 1;
            e.b0 -= 1;
        }
        while e.a1 < n && e.b1 < m && a[e.a1] == b[e.b1] && word(a[e.a1]) && touches_word(e, false) {
            e.a1 += 1;
            e.b1 += 1;
        }
        // Chinese: a one-character fix needs a neighbour to mean anything ("长"→"章" is
        // learned as "成长"→"成章")
        let wrong = &a[e.a0..e.a1];
        if !wrong.iter().any(|&c| word(c)) && wrong.iter().filter(|&&c| is_cjk(c)).count() == 1 {
            if e.a0 > 0 && e.b0 > 0 && a[e.a0 - 1] == b[e.b0 - 1] && is_cjk(a[e.a0 - 1]) {
                e.a0 -= 1;
                e.b0 -= 1;
                e.grown = true;
            } else if e.a1 < n && e.b1 < m && a[e.a1] == b[e.b1] && is_cjk(a[e.a1]) {
                e.a1 += 1;
                e.b1 += 1;
                e.grown = true;
            }
        }
    }
    let edits = merge(edits, &|_, _| false);

    let mut pairs: Vec<Pair> = vec![];
    let mut terms: Vec<String> = vec![];
    for e in &edits {
        let wrong = trim_side(&a[e.a0..e.a1].iter().collect::<String>());
        let right = trim_side(&b[e.b0..e.b1].iter().collect::<String>());
        let (cw, cr) = (core(&wrong), core(&right));
        if cw.is_empty() || cr.is_empty() || cw == cr {
            continue; // added or removed words, or punctuation only
        }
        let (lw, lr) = (wrong.chars().count(), right.chars().count());
        if lw > MAX_SIDE || lr > MAX_SIDE || (lw.max(lr) > 2 * lw.min(lr) + 2 && !has_ascii_word(&right)) {
            continue; // a rewrite, not a misheard word
        }
        if !pairs.iter().any(|(w, _)| *w == wrong) {
            pairs.push((wrong.clone(), right.clone()));
        }
        for t in term_candidates(&wrong, &right, e.grown) {
            if !terms.contains(&t) {
                terms.push(t);
            }
        }
    }
    Some(Learned { corrected, changed, pairs, terms })
}

/// The corrected spelling, when it looks like a name or a term: English words that
/// weren't already there ("Redis", "PostgreSQL"), or a Chinese fix of two or more
/// characters ("张三丰"). A one-character Chinese fix ("周二"→"周三") isn't a term.
fn term_candidates(wrong: &str, right: &str, grown: bool) -> Vec<String> {
    let mut out = vec![];
    if has_ascii_word(right) {
        let keep = |c: char| word(c) || " +#.-_".contains(c);
        for run in right.split(|c: char| !keep(c)) {
            let run = run.trim_matches(|c: char| c == ' ' || c == '.' || c == '-' || c == '_');
            if run.chars().count() >= 2 && run.chars().any(|c| c.is_ascii_alphabetic()) && !wrong.contains(run) {
                out.push(run.to_string());
            }
        }
    } else if !grown {
        let n = right.chars().count();
        if (2..=8).contains(&n) && right.chars().all(|c| is_cjk(c) || c.is_alphanumeric()) {
            out.push(right.to_string());
        }
    }
    out
}

// ----------------------------------------------------------------------------- applying

/// A pair whose wrong side has English in it is unambiguous ("read is" is never what the
/// user meant), so it's replaced directly; a Chinese one ("成长"→"成章") depends on the
/// sentence and is left to the polishing model.
pub fn is_literal(p: &Pair) -> bool {
    has_ascii_word(&p.0)
}

/// Replace learned mistakes in a transcript, in one left-to-right pass (no cascades).
/// English is matched case-insensitively on word boundaries. `chinese_too` also replaces
/// Chinese pairs — for when no polishing model is there to judge them.
pub fn apply(text: &str, pairs: &[Pair], chinese_too: bool) -> String {
    let mut rules: Vec<(String, &str, bool)> = pairs
        .iter()
        .filter(|p| !p.0.is_empty() && (chinese_too || is_literal(p)))
        .map(|p| (p.0.to_ascii_lowercase(), p.1.as_str(), is_literal(p)))
        .collect();
    if rules.is_empty() {
        return text.to_string();
    }
    rules.sort_by_key(|r| std::cmp::Reverse(r.0.len()));
    let low = text.to_ascii_lowercase(); // same byte offsets as `text`
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    'scan: while i < text.len() {
        for (w, r, literal) in &rules {
            if !low[i..].starts_with(w.as_str()) {
                continue;
            }
            if *literal {
                let before = text[..i].chars().next_back();
                let after = text[i + w.len()..].chars().next();
                let wc = |c: Option<char>| c.is_some_and(word);
                if (wc(w.chars().next()) && wc(before)) || (wc(w.chars().next_back()) && wc(after)) {
                    continue; // inside a longer word ("rust" in "trust")
                }
            }
            out.push_str(r);
            i += w.len();
            continue 'scan;
        }
        let c = text[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// The transcript contains a learned Chinese mistake: worth a pass through the
/// polishing model even when it's short enough for the fast path.
pub fn mentions(text: &str, pairs: &[Pair]) -> bool {
    pairs.iter().any(|p| !is_literal(p) && !p.0.is_empty() && text.contains(p.0.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(l: &Learned) -> Vec<(&str, &str)> {
        l.pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
    }

    #[test]
    fn english_terms_become_whole_words() {
        let l = learn("我们后端用的是 post gray sql，缓存用的是 read is。", "我们后端用的是 PostgreSQL，缓存用的是 Redis。").unwrap();
        assert_eq!(pairs(&l), vec![("post gray sql", "PostgreSQL"), ("read is", "Redis")]);
        assert_eq!(l.terms, vec!["PostgreSQL", "Redis"]);
    }

    #[test]
    fn case_fix_widens_to_the_word() {
        let l = learn("这个项目是用 rust 写的。", "这个项目是用 Rust 写的。").unwrap();
        assert_eq!(pairs(&l), vec![("rust", "Rust")]);
        assert_eq!(l.terms, vec!["Rust"]);
    }

    #[test]
    fn one_chinese_char_takes_a_neighbour_and_is_no_term() {
        let l = learn("我们的项目叫成长，下周发布。", "我们的项目叫成章，下周发布。").unwrap();
        assert_eq!(pairs(&l), vec![("成长", "成章")]);
        assert!(l.terms.is_empty());
    }

    #[test]
    fn longer_chinese_fix_is_a_term() {
        let l = learn("刚才张山峰说要改需求。", "刚才张三丰说要改需求。").unwrap();
        assert_eq!(pairs(&l), vec![("山峰", "三丰")]);
        assert_eq!(l.terms, vec!["三丰"]);
    }

    #[test]
    fn surrounding_text_and_punctuation_are_ignored() {
        let l = learn("我们周三开会", "Hi 各位：\n我们周三开会。\n谢谢").unwrap();
        assert_eq!(l.corrected, "我们周三开会");
        assert!(l.pairs.is_empty());
        let l = learn("我们周三开会在三楼", "我们周三开会，在三楼").unwrap();
        assert!(l.changed && l.pairs.is_empty());
    }

    #[test]
    fn partial_selection() {
        let l = learn("缓存用的是 read is，数据库是 my sequel。", "缓存用的是 Redis").unwrap();
        assert_eq!(pairs(&l), vec![("read is", "Redis")]);
        assert_eq!(l.corrected, "缓存用的是 Redis");
    }

    #[test]
    fn insertions_deletions_and_rewrites_are_not_pairs() {
        let l = learn("我觉得这个设计还可以", "我觉得这个设计还可以再简单一点").unwrap();
        assert!(l.pairs.is_empty());
        let l = learn("嗯我觉得这个设计还可以", "我觉得这个设计还可以").unwrap();
        assert!(l.pairs.is_empty());
        // a selection that ends inside a fix doesn't make "长，下周发布" a misrecognition of "章"
        let l = learn("我们的项目叫成长，下周发布。", "成章").unwrap();
        assert!(l.pairs.is_empty());
    }

    #[test]
    fn no_change() {
        let l = learn("你好", "你好").unwrap();
        assert!(!l.changed && l.pairs.is_empty());
    }

    #[test]
    fn finds_the_dictation_inside_a_long_document() {
        let filler = "这是一段很长的文档内容，里面有各种各样的文字。".repeat(40);
        let doc = format!("{filler}我们用 Redis 做缓存。{filler}");
        assert!(similarity("我们用 read is 做缓存。", &doc) > 0.6);
        let l = learn("我们用 read is 做缓存。", &doc).unwrap();
        assert_eq!(pairs(&l), vec![("read is", "Redis")]);
        assert!(similarity("完全不相关的一句话", "我们用 Redis 做缓存") < 0.3);
        assert!(similarity("缓存用的是 read is，数据库是 my sequel。", "缓存用的是 Redis") > 0.6);
    }

    #[test]
    fn apply_replaces_english_on_word_boundaries_only() {
        let p = vec![("read is".to_string(), "Redis".to_string()), ("rust".to_string(), "Rust".to_string()),
                     ("成长".to_string(), "成章".to_string())];
        assert_eq!(apply("缓存用Read is，语言用rust", &p, false), "缓存用Redis，语言用Rust");
        assert_eq!(apply("I trust it, 我在成长", &p, false), "I trust it, 我在成长");
        assert_eq!(apply("我在成长", &p, true), "我在成章");
        assert!(mentions("我们的成长", &p) && !mentions("用 read is", &p));
    }

    #[test]
    fn apply_does_not_cascade() {
        let p = vec![("a b".to_string(), "c".to_string()), ("c".to_string(), "d".to_string())];
        assert_eq!(apply("a b c", &p, false), "c d");
    }
}
