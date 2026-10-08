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
    assert "crackles" in paragraphs[2].casefold()
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


def test_radiology_report_keeps_indication_findings_impression_and_recommendation() -> None:
    source = """\
Clinical Indication
The patient has had a worsening cough and fever for five days.
Imaging Findings
CT scan shows bilateral patchy opacities in the lower lungs.
Impression
The findings are consistent with pneumonia.
Recommendations
Repeat chest imaging in six weeks to confirm the opacities have resolved.
"""

    summary = summarize_text(source).casefold()

    assert "worsening cough and fever for five days" in summary
    assert "bilateral (on both sides) patchy opacities" in summary
    assert "consistent with pneumonia" in summary
    assert "repeat chest imaging in six weeks" in summary
    assert "cardiac" not in summary


def test_pathology_and_laboratory_reports_include_results_and_interpretation() -> None:
    pathology = """\
Clinical History
The patient has a skin lesion that has increased in size over three months.
Specimen
A biopsy was taken from the left forearm.
Microscopic Description
The specimen contains atypical cells in the sampled tissue.
Final Diagnosis
The findings are consistent with a malignant melanoma.
"""
    laboratory = """\
Clinical Information
The patient reports fatigue and dizziness.
Laboratory Results
Hemoglobin is low at 8.2 g/dL and the complete blood count shows anemia.
Interpretation
The results indicate anemia.
Plan
Repeat laboratory testing in two weeks.
"""

    pathology_summary = summarize_text(pathology).casefold()
    laboratory_summary = summarize_text(laboratory).casefold()

    assert "skin lesion" in pathology_summary
    assert "biopsy" in pathology_summary
    assert "malignant (cancerous) melanoma" in pathology_summary
    assert "fatigue and dizziness" in laboratory_summary
    assert "hemoglobin is low at 8.2 g/dl" in laboratory_summary
    assert "anemia" in laboratory_summary
    assert "repeat laboratory testing in two weeks" in laboratory_summary


def test_discharge_summary_preserves_course_diagnosis_and_follow_up() -> None:
    source = """\
Reason for Admission
The patient was admitted with dehydration after three days of vomiting.
Hospital Course
Symptoms improved after intravenous fluids, and the patient is tolerating oral fluids.
Discharge Diagnosis
The final diagnosis is acute gastroenteritis with dehydration.
Discharge Instructions
Continue oral fluids and return for worsening symptoms or reduced urination.
Follow-up
See the primary care clinician in one week.
"""

    summary = summarize_text(source).casefold()

    assert "three days of vomiting" in summary
    assert "symptoms improved" in summary
    assert "dehydration" in summary
    assert "return for worsening symptoms" in summary
    assert "primary care clinician in one week" in summary


def test_ocr_inline_section_headings_are_recognized() -> None:
    source = """\
History of Present Illness
The patient reports a persistent cough.
Surgical � 2020: Appendectomy Medical History � Diagnosed with asthma in childhood.
Physical Examination
Wheezing is heard in both lungs.
Impression
The findings are consistent with an asthma exacerbation.
Plan
Continue the prescribed inhaler and return if breathing worsens.
"""

    summary = summarize_text(source).casefold()

    assert "appendectomy" in summary
    assert "asthma in childhood" in summary
    assert "wheezing is heard" in summary
    assert "asthma exacerbation" in summary


def test_clinical_comment_is_kept_but_tutorial_comment_is_removed() -> None:
    result = summarize_text("""\
Comment: Define the reason for the patient's visit as specifically as possible.
The patient presented to the emergency department with abdominal pain.
Comment: No acute fracture is identified on the X-ray.
The patient may return to normal activity as tolerated.
""")

    assert "define the reason" not in result.casefold()
    assert "abdominal pain" in result.casefold()
    assert "no acute fracture" in result.casefold()
    assert "return to normal activity" in result.casefold()
