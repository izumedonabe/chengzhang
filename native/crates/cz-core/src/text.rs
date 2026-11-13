//! Small text helpers shared by the engine (ported from engine.py).

pub const FILLERS: &[&str] = &["嗯", "呃", "啊", "额", "那个", "就是", "然后", "不对", "不是", "对吧"];

/// Letters and digits only, lowercased, full-width ASCII folded to half-width
/// (what NFKC does for the cases that occur in dictation).
pub fn norm(s: &str) -> String {
    s.chars()
        .map(|c| match c as u32 {
            0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .flat_map(|c| c.to_lowercase())
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// True when some `span`-char chunk repeats `times`+ times — a decoding loop.
pub fn looping(text: &str, span: usize, times: usize) -> bool {
    let t: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if t.len() < span * times {
        return false;
    }
    let mut seen = std::collections::HashMap::<&[char], usize>::new();
    for i in 0..=t.len() - span {
        let n = seen.entry(&t[i..i + span]).or_insert(0);
        *n += 1;
        if *n >= times {
            return true;
        }
    }
    false
}

fn ascii_alnum(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_alphanumeric())
}

/// Join segment transcripts, adding a space only between two ASCII words.
pub fn join_segments<S: AsRef<str>>(texts: &[S]) -> String {
    let mut out = String::new();
    for t in texts {
        let t = t.as_ref();
        if ascii_alnum(out.chars().last()) && ascii_alnum(t.chars().next()) {
            out.push(' ');
        }
        out += t;
    }
    out
}

/// Short, filler-free utterances skip the LLM.
pub fn needs_llm(raw: &str, mode: Option<&str>, fast_path: bool) -> bool {
    if mode.is_some() || !fast_path {
        return true;
    }
    norm(raw).chars().count() > 12 || FILLERS.iter().any(|f| raw.contains(f))
}

/// Output much shorter/longer than the input probably isn't a cleanup.
pub fn suspicious(raw: &str, out: &str, mode: Option<&str>) -> bool {
    if looping(out, 8, 4) {
        return true;
    }
    if mode.is_some() || raw.contains("翻译") {
        return out.is_empty();
    }
    let r = norm(raw).chars().count() as f32;
    let o = norm(out).chars().count() as f32;
    o == 0.0 || o < 0.35 * r || o > 1.8 * r + 20.0 || answered(raw, out)
}

/// Cleaning up trims and rearranges what was said; answering a question ("为什么天空是蓝色的"
/// → "天空呈现蓝色是因为瑞利散射…") brings in a lot of wording that was never said.
/// Digits don't count: "二零二六" legitimately becomes "2026".
fn answered(raw: &str, out: &str) -> bool {
    let mut have: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for c in norm(raw).chars() {
        *have.entry(c).or_default() += 1;
    }
    let (mut new, mut total) = (0usize, 0usize);
    for c in norm(out).chars().filter(|c| !c.is_ascii_digit()) {
        total += 1;
        match have.get_mut(&c) {
            Some(n) if *n > 0 => *n -= 1,
            _ => new += 1,
        }
    }
    new >= 5 && new as f32 > 0.4 * total as f32
}

/// Qwen3-ASR gets the personal dictionary as context; on unclear audio (noise, a talk
/// heard from across the room) it can "transcribe" that context back verbatim:
/// "OpenTypeLess, database, README, …". Remove runs that copy most of the list in its
/// own order. A sentence that merely uses a few dictionary words is left alone.
pub fn strip_context_echo(text: &str, dict: &[String]) -> String {
    let words: Vec<&str> = dict.iter().map(|w| w.trim()).filter(|w| !w.is_empty()).collect();
    if words.len() < 3 {
        return text.to_string();
    }
    let need = 3.max((words.len() * 3 + 4) / 5); // >= 60% of the list, at least 3
    let is_sep = |c: char| matches!(c, ',' | '，' | '、' | ';' | '；' | ' ' | '\u{3000}');
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        // try to read words[k], words[k+1], … starting here
        let mut best: Option<usize> = None;
        for start in 0..words.len() {
            let (mut j, mut n, mut k) = (i, 0, start);
            while k < words.len() {
                let mut p = j;
                if n > 0 {
                    while let Some(c) = text[p..].chars().next().filter(|&c| is_sep(c)) {
                        p += c.len_utf8();
                    }
                }
                if !text[p..].starts_with(words[k]) {
                    break;
                }
                j = p + words[k].len();
                n += 1;
                k += 1;
            }
            if n >= need && best.map_or(true, |b| j > b) {
                best = Some(j);
            }
        }
        if let Some(mut j) = best {
            // drop the separator / full stop that followed the echo
            while let Some(c) = text[j..].chars().next().filter(|&c| is_sep(c) || matches!(c, '。' | '.')) {
                j += c.len_utf8();
            }
            i = j;
            continue;
        }
        let c = text[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_echo() {
        let d: Vec<String> = ["OpenTypeLess", "database", "README", "UI稿", "fail to pass", "pass to pass", "Qwen"]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(strip_context_echo("OpenTypeLess, database, README, UI稿, fail to pass, pass to pass, Qwen", &d), "");
        assert_eq!(
            strip_context_echo("说：“你的是平常。”OpenTypeLess, database, README, UI稿, fail to pass, pass to pass, Qwen嗯，应该挺好的。", &d),
            "说：“你的是平常。”嗯，应该挺好的。"
        );
        // ordinary use of a few dictionary words stays
        let s = "把 README 和 database 的 UI稿 发给我";
        assert_eq!(strip_context_echo(s, &d), s);
        assert_eq!(strip_context_echo("OpenTypeLess、database、README 都要改", &d), "OpenTypeLess、database、README 都要改");
    }

    #[test]
    fn basics() {
        assert_eq!(norm("Ａb，１2。"), "ab12");
        assert_eq!(join_segments(&["hello", "world", "你好"]), "hello world你好");
        assert!(looping(&"一二三四五六七八".repeat(4), 8, 4));
        assert!(!looping("今天天气不错", 8, 4));
        assert!(!needs_llm("好的", None, true));
        assert!(needs_llm("嗯好的", None, true));
        assert!(suspicious("我们明天开会讨论一下这个方案", "好", None));
        assert!(!suspicious("我们明天开会讨论一下这个方案", "我们明天开会讨论这个方案。", None));
    }

    #[test]
    fn answers_are_suspicious_cleanups_are_not() {
        // answers
        assert!(suspicious("为什么天空是蓝色的", "天空呈现蓝色是因为瑞利散射，蓝光更容易被空气分子散射。", None));
        assert!(suspicious("请问一下北京今天天气怎么样", "北京今天晴，最高气温十五度。", None));
        // cleanups
        assert!(!suspicious("嗯那个就是我想问一下咱们这个项目什么时候上线", "我想问一下咱们这个项目什么时候上线？", None));
        assert!(!suspicious("我们后端用的是post gray sql然后缓存用的是read is", "我们后端用的是 PostgreSQL，缓存用的是 Redis。", None));
        assert!(!suspicious("这个千万模型挺好的", "这个 Qwen 模型挺好的。", None));
        assert!(!suspicious("我们二零二六年的目标是用户增长百分之十五", "我们 2026 年的目标是用户增长 15%。", None));
        assert!(!suspicious("那个Docker和虚拟机到底有什么区别啊", "Docker 和虚拟机到底有什么区别？", None));
    }
}
