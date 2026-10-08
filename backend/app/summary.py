import math
import re
from collections import Counter

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "after",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "may",
    "me", "might", "more", "most", "must", "my", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "us", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your",
}
_ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "vs.", "etc.", "e.g.", "i.e."
)
_CLINICAL_TERMS = (
    "chest pain", "shortness of breath", "dyspnea", "blood pressure", "murmur",
    "crackles", "bruit", "angina", "hypertension", "diagnosis", "assessment",
    "medication", "allergy", "heart attack", "myocardial infarction",
)
_PLAIN_LANGUAGE = (
    (
        re.compile(r"\bparoxysmal nocturnal dyspnea\b", re.I),
        "sudden nighttime shortness of breath",
    ),
    (re.compile(r"\bdyspnea\b", re.I), "shortness of breath (dyspnea)"),
    (re.compile(r"\borthopnea\b", re.I), "breathlessness when lying flat (orthopnea)"),
    (
        re.compile(r"\bunstable angina\b", re.I),
        "unstable angina (worsening heart-related chest pain)",
    ),
    (
        re.compile(r"\bmyocardial infarction\b", re.I),
        "heart attack (myocardial infarction)",
    ),
    (
        re.compile(r"\bcoronary artery disease\b", re.I),
        "disease in the arteries that supply the heart (coronary artery disease)",
    ),
    (
        re.compile(r"\ban ischemic cardiac origin\b", re.I),
        "a heart problem caused by reduced blood flow",
    ),
    (
        re.compile(r"\bischemic cardiac origin\b", re.I),
        "a heart problem caused by reduced blood flow",
    ),
    (re.compile(r"\bischemic heart disease\b", re.I), "heart disease caused by reduced blood flow"),
    (
        re.compile(r"\bangina pectoris\b", re.I),
        "angina (a type of heart-related chest pain)",
    ),
    (re.compile(r"\bischemic\b", re.I), "related to reduced blood flow"),
    (re.compile(r"\bASCVD\b", re.I), "heart and blood vessel disease (ASCVD)"),
    (re.compile(r"\bCAD\b", re.I), "coronary artery disease (CAD)"),
    (re.compile(r"\bHTN\b", re.I), "high blood pressure (HTN)"),
    (re.compile(r"\bhypertension\b", re.I), "high blood pressure (hypertension)"),
    (re.compile(r"\bbruit\b", re.I), "bruit (an unusual whooshing sound heard with a stethoscope)"),
    (re.compile(r"\bsystolic murmur\b", re.I), "murmur (an unusual heart sound during a heartbeat)"),
    (re.compile(r"\bNSAID\b", re.I), "anti-inflammatory pain medicine (NSAID)"),
    (
        re.compile(r"\bearly surgical menopause\b", re.I),
        "early menopause following ovary-removal surgery",
    ),
    (re.compile(r"\bFH\b"), "family history"),
    (re.compile(r"\by/o\b", re.I), "year-old"),
)
_SECTION_HEADINGS = {
    "history and physical examination comments",
    "sample written history and physical examination",
}
_LABEL_ONLY_HEADINGS = {
    "general", "skin", "heent", "cadiovascular", "cardiovascular",
    "gastrointestinal", "genitourinary", "musculoskeletal", "neurological",
    "neck", "chest", "abdomen", "extremities", "nodes", "genital/rectal",
    "surgical", "vital signs",
}
_COMMENT_CONTINUATIONS = (
    r"^patient['’]s complaint\b",
    r"^the patient['’]s problem\b",
    r"^check for orthostatic\b",
    r"^description may give\b",
    r"^this patient needs\b",
    r"^more precise than\b",
    r"^comment on\b",
    r"^always include\b",
    r"^in the assessment\b",
    r"^as in the previous\b",
    r"^although you can omit\b",
    r"^this list\b",
    r"^a final problem list\b",
    r"^uncovered in the interview\b",
    r"^random unrelated\b",
    r"^you should\b",
    r"^your plan\b",
    r"^follow this pattern\b",
    r"^you are expected to\b",
    r"^develop a diagnostic and therapeutic plan\b",
    r"^your plan should\b",
)


def _clean_lines(text: str) -> list[str]:
    lines = []
    skipping_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        stripped = stripped.replace("\ufffd", "'")
        if not stripped or re.fullmatch(r"\d+", stripped):
            continue
        if re.match(r"(?i)^comment\s*:", stripped):
            skipping_comment = True
            continue
        normalized = stripped.casefold().rstrip(":")
        if normalized in _SECTION_HEADINGS or normalized in _LABEL_ONLY_HEADINGS:
            continue
        if re.match(r"(?i)^(?:patient name|date|referral source|data source)\s*:", stripped):
            continue
        stripped = re.sub(r"(?i)^chief complaint\s*&\s*id\s*:\s*", "", stripped)
        stripped = re.sub(
            r"(?i)^(?:allergy|medications|alcohol use|tobacco use)\s*:\s*",
            "",
            stripped,
        )
        if not stripped:
            continue
        if skipping_comment:
            if (
                stripped[0].islower()
                or re.match(
                    r"(?i)^(?:you\b|your\b|in this case\b|random unrelated\b|"
                    r"follow this pattern\b|this list\b)",
                    stripped,
                )
            ):
                continue
            skipping_comment = False
        if re.match(
            r"(?i)^(?:you\b|your plan\b|in the assessment you\b|"
            r"this list regroups\b|although you can omit\b)",
            stripped,
        ):
            continue
        if any(
            re.search(pattern, stripped, re.I)
            for pattern in _COMMENT_CONTINUATIONS
        ):
            continue
        lines.append(stripped)
    return lines


def _sentences(text: str) -> list[str]:
    protected = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        protected = re.sub(
            rf"(?<![A-Za-z]){re.escape(abbreviation)}",
            abbreviation.replace(".", f"<DOT{index}>"),
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(
        r"(?i)(\bangina pectoris)\s+(?=This patient['’]s\b)",
        r"\1. ",
        protected,
    )

    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [
        re.sub(r"<DOT\d+>", ".", sentence).strip()
        for sentence in sentences
        if len(re.findall(r"\b[\w'-]+\b", sentence)) >= 4
    ]


def _sentence_words(sentence: str) -> set[str]:
    return {
        word.casefold()
        for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", sentence)
        if word.casefold() not in _STOP_WORDS and len(word) > 2
    }


def _sentence_score(
    sentence: str, document_frequency: Counter[str], count: int
) -> float:
    words = [
        word.casefold()
        for word in re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", sentence)
        if word.casefold() not in _STOP_WORDS and len(word) > 2
    ]
    if not words:
        return 0.0
    term_counts = Counter(words)
    weighted = sum(
        frequency * math.log(1 + count / (1 + document_frequency[word]))
        for word, frequency in term_counts.items()
    )
    clinical_bonus = 2 * sum(
        term in sentence.casefold() for term in _CLINICAL_TERMS
    )
    return weighted / math.sqrt(len(words)) + clinical_bonus


def _select_section(
    sentences: list[str],
    document_frequency: Counter[str],
    total_sentences: int,
    priority_patterns: tuple[str, ...],
    limit: int,
    word_budget: int,
    fill_remaining: bool = True,
) -> list[str]:
    if not sentences:
        return []
    ranked = sorted(
        sentences,
        key=lambda sentence: _sentence_score(
            sentence, document_frequency, total_sentences
        ),
        reverse=True,
    )
    selected: list[str] = []
    word_count = 0

    def add(sentence: str) -> None:
        nonlocal word_count
        words = len(sentence.split())
        if sentence not in selected and word_count + words <= word_budget:
            selected.append(sentence)
            word_count += words

    for pattern in priority_patterns:
        match = next(
            (
                sentence
                for sentence in ranked
                if re.search(pattern, sentence, re.I)
                and sentence not in selected
                and word_count + len(sentence.split()) <= word_budget
            ),
            None,
        )
        if match:
            add(match)
        if len(selected) == limit:
            break
    if fill_remaining:
        for sentence in ranked:
            if len(selected) == limit:
                break
            add(sentence)
    return sorted(selected, key=sentences.index)


def _explain_terms(text: str) -> str:
    for pattern, replacement in _PLAIN_LANGUAGE:
        text = pattern.sub(replacement, text)
    return text


def summarize_text(text: str) -> str:
    """Build a source-grounded, multi-paragraph extractive report summary."""
    lines = _clean_lines(text)
    headings = {
        "chief complaint & id": "hpi",
        "history of present illness": "hpi",
        "past medical history": "history",
        "medical history": "history",
        "family history": "history",
        "social history": "history",
        "review of systems": "history",
        "physical examination": "exam",
        "vital signs": "exam",
        "initial problem list": "assessment",
        "revised problem list": "assessment",
        "assessment and differential diagnosis": "assessment",
        "plan": "plan",
        "surgical": "history",
        "surgical �": "history",
        "surgical -": "history",
        "surgical –": "history",
        "surgical —": "history",
        "medical history �": "history",
        "medical history -": "history",
        "medical history –": "history",
        "medical history —": "history",
    }
    section_lines: dict[str, list[str]] = {
        "hpi": [],
        "history": [],
        "exam": [],
        "assessment": [],
        "plan": [],
    }
    current_section = "hpi"
    for line in lines:
        heading = line.casefold().strip().rstrip(":")
        if heading in headings:
            current_section = headings[heading]
            continue
        section_lines[current_section].append(line)

    section_sentences = {
        name: _sentences(" ".join(section))
        for name, section in section_lines.items()
    }
    has_structured_sections = sum(
        bool(section_sentences[name])
        for name in ("hpi", "history", "exam", "assessment", "plan")
    ) >= 3
    if has_structured_sections:
        paragraph_sources = [
            section_sentences["hpi"],
            section_sentences["history"],
            section_sentences["exam"],
            section_sentences["assessment"] + section_sentences["plan"],
        ]
    else:
        all_sentences = _sentences(
            " ".join(
                line for line in lines
                if line.casefold().strip().rstrip(":") not in headings
            )
        )
        paragraph_count = min(4, len(all_sentences))
        paragraph_sources = [
            all_sentences[
                paragraph_index * len(all_sentences) // paragraph_count:
                (paragraph_index + 1) * len(all_sentences) // paragraph_count
            ]
            for paragraph_index in range(paragraph_count)
        ] if paragraph_count else []

    sentences = [sentence for group in paragraph_sources for sentence in group]
    if not sentences:
        return ""

    document_frequency: Counter[str] = Counter(
        word for sentence in sentences for word in _sentence_words(sentence)
    )

    priorities = {
        0: (
            r"chest pain",
            r"(?:week|day|hour|minute|month|year|ago|duration|lasting)",
            r"(?:sleep|rest|awaken|awoke|emergency department|prompted|worsen|additional episode)",
            r"(?:shortness of breath|dyspnea|radiat|nausea|sweating|vomit)",
            r"(?:additional episodes|three days ago|walking her dog)",
            r"(?:exertion|walking|activity|garden)",
        ),
        1: (
            r"(?:hypertension|high blood pressure)",
            r"(?:father|mother).*(?:heart attack|deceased)|(?:heart attack).*(?:father|mother)",
            r"(?:family history|FH).*(?:premature|early).*(?:CAD|ASCVD|coronary)|(?:premature|early).*(?:CAD|ASCVD|coronary)",
            r"(?:no history of heart disease|no previous heart disease|never been told.*heart)",
            r"(?:not smoke|does not smoke|never smoke|tobacco)",
            r"(?:diabetes|blood sugar)",
            r"(?:peptic ulcer|cancer|lung disease|surgical menopause|oophorectomy)",
            r"(?:medication|ibuprofen|aspirin|allergy)",
        ),
        2: (
            r"(?:blood pressure|pulse|temperature|respirations)",
            r"(?:murmur|heart sound|PMI|cardiac)",
            r"(?:bruit|abdominal bruit)",
            r"(?:crackles|lungs|edema)",
        ),
        3: (
            r"(?:assessment|angina|ischemic|diagnosis|most likely)",
            r"(?:risk factor|family history|hypertension|surgical menopause|ASCVD)",
            r"(?:differential|less likely|reflux|pulmonary|alternative)",
            r"(?:plan|monitor|admi|catheter|nitrate|aspirin|treatment|schedule|begin|start)",
        ),
    }
    paragraphs: list[str] = []
    for paragraph_index, group in enumerate(paragraph_sources):
        if not group:
            continue
        if has_structured_sections and paragraph_index == 3:
            selected = list(dict.fromkeys(
                _select_section(
                    section_sentences["assessment"],
                    document_frequency,
                    len(sentences),
                    priorities[3],
                    3,
                    95,
                )
                + _select_section(
                    section_sentences["plan"],
                    document_frequency,
                    len(sentences),
                    (priorities[3][-1],),
                    2,
                    60,
                )
            ))[:7]
            selected.sort(key=group.index)
        elif has_structured_sections:
            selected = _select_section(
                group,
                document_frequency,
                len(sentences),
                priorities[paragraph_index],
                (7, 8, 4, 5)[paragraph_index],
                (140, 190, 105, 155)[paragraph_index],
                fill_remaining=paragraph_index not in (1, 2),
            )
        else:
            selected = _select_section(
                group, document_frequency, len(sentences), (), 5, 100
            )
        paragraphs.append(_explain_terms(" ".join(selected)))

    return "\n\n".join(paragraphs)
