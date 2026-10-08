from app.summary import _sentences, summarize_text


def test_summary_contains_multiple_paragraphs_from_source() -> None:
    source_sentences = [
        "The patient reports chest pain during activity and shortness of breath.",
        "The pain began one week ago and radiates to her neck.",
        "She has a history of hypertension and peptic ulcer disease.",
        "Her father died of a heart attack at age fifty-four.",
        "Blood pressure is 168 over 98 and pulse is 90.",
        "Crackles are heard at both lung bases.",
        "The assessment is chest pain with features of angina.",
        "The report notes dyspnea and recent onset hypertension.",
    ]
    summary = summarize_text(" ".join(source_sentences))
    paragraphs = summary.split("\n\n")

    assert len(paragraphs) == 4
    assert all(paragraph.strip() for paragraph in paragraphs)
    normalized_summary = summary.casefold()
    assert "chest pain" in normalized_summary
    assert "one week ago" in normalized_summary
    assert "high blood pressure (hypertension)" in normalized_summary
    assert "father died of a heart attack" in normalized_summary
    assert "blood pressure is 168 over 98" in normalized_summary
    assert "crackles are heard" in normalized_summary
    assert "features of angina" in normalized_summary
    assert "shortness of breath" in normalized_summary


def test_summary_does_not_invent_facts() -> None:
    source = (
        "The patient reports chest pain during activity and shortness of breath. "
        "The patient has no history of heart disease. Her blood pressure is elevated."
    )

    result = summarize_text(source)

    assert "no history of heart disease" in result
    assert "chronic obstructive sleep apnea" not in result


def test_summary_omits_ocr_tutorial_comments_and_page_numbers() -> None:
    source = """\
Patient reports chest pain during activity and shortness of breath.
Comment: Always list vital signs.
patient's complaint and documented results.
2
Blood pressure is 168 over 98 and pulse is 90.
History and Physical Examination Comments
"""

    result = summarize_text(source)

    assert "Comment:" not in result
    assert "Always list vital signs" not in result
    assert "History and Physical Examination Comments" not in result
    assert "2" not in result
    assert "Blood pressure is 168 over 98 and pulse is 90." in result


def test_summary_omits_document_headings_and_patient_metadata() -> None:
    source = """\
Patient Name: Sample Patient
Date: 1/2/2025
Chief Complaint & ID: The patient reports chest pain during activity.
History of Present Illness
The pain began one week ago and radiates to her neck.
Assessment and Differential Diagnosis
The assessment is chest pain with features of angina.
"""

    result = summarize_text(source)

    assert "Patient Name" not in result
    assert "History of Present Illness" not in result
    assert "Assessment and Differential Diagnosis" not in result
    assert "The patient reports chest pain during activity." in result


def test_structured_report_summary_covers_assessment_and_plan() -> None:
    source = """\
History of Present Illness
The patient reports chest pain during activity and shortness of breath.
The pain began one week ago and radiates to her neck.
Past Medical History
She has a history of hypertension and peptic ulcer disease.
Her father died of a heart attack at age fifty-four.
Physical Examination
A systolic murmur radiates to the neck.
Crackles are heard at both lung bases.
Assessment and Differential Diagnosis
The assessment is chest pain with features of angina.
Plan
The documented plan includes monitoring and cardiac evaluation.
"""

    paragraphs = summarize_text(source).split("\n\n")

    assert len(paragraphs) == 4
    assert "chest pain" in paragraphs[0]
    assert "hypertension" in paragraphs[1]
    assert "murmur" in paragraphs[2]
    assert "angina" in paragraphs[3]
    assert "monitoring and cardiac evaluation" in paragraphs[3]


def test_summary_covers_progression_risk_factors_and_plain_language() -> None:
    source = """\
History of Present Illness
The patient is a 56 year old woman with one week of dull chest pain radiating to her neck.
She has shortness of breath during the episodes.
The latest episode awakened her from sleep and lasted 30 minutes, prompting an Emergency Department visit.
Past Medical History
She was diagnosed with hypertension three years ago.
She has no history of previous heart disease and does not smoke or have diabetes.
Her father died of a heart attack at age 54, and there is a family history of premature ASCVD.
She had a hysterectomy and removal of both ovaries at age 44.
Physical Examination
Blood pressure is 168/98 and pulse is 90.
Crackles are heard at both lung bases.
A systolic murmur radiates to the neck.
An abdominal bruit is heard on the right.
Assessment and Differential Diagnosis
Chest pain with features of angina pectoris is the most likely diagnosis.
The patient's exertional chest pain and risk factors suggest an ischemic cardiac origin.
The latest episode at rest suggests unstable angina, and hospitalization is indicated.
Plan
Admit the patient for monitoring and schedule cardiac catheterization.
Start aspirin and nitrates to reduce the risk of myocardial infarction and treat pain.
"""

    paragraphs = summarize_text(source).split("\n\n")
    summary = " ".join(paragraphs)

    assert len(paragraphs) == 4
    assert "awakened her from sleep and lasted 30 minutes" in summary
    assert "hypertension" in summary
    assert "does not smoke" in summary.casefold()
    assert "have diabetes" in summary.casefold()
    assert "father died of a heart attack" in summary
    assert "premature heart and blood vessel disease (ASCVD)" in summary
    assert "blood pressure is 168/98" in summary.casefold()
    assert "crackles are heard" in summary.casefold()
    assert "murmur (an unusual heart sound" in summary
    assert "bruit (an unusual whooshing sound" in summary
    assert "angina (a type of heart-related chest pain)" in summary
    assert "unstable angina" in summary.casefold()
    assert "cardiac catheterization" in summary.casefold()


def test_empty_or_non_sentence_text_returns_empty_summary() -> None:
    assert summarize_text(" \n 2 \n Comment: remove this") == ""


def test_sentence_segmentation_keeps_common_name_abbreviations() -> None:
    sentences = _sentences(
        "Ms. Rogers reports chest pain during activity and shortness of breath. "
        "Her blood pressure is elevated at the clinic today."
    )

    assert len(sentences) == 2
    assert sentences[0].startswith("Ms. Rogers")
