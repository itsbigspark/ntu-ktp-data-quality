"""
Tests for normalized fuzzy correction and email-domain correction
(core/validator/corpus_correction.py).
"""

from core.validator.corpus_correction import suggest_fix, suggest_email_fix


BANK_CORPUS = ["NatWest", "Metro Bank", "Barclays", "HSBC UK", "Lloyds Banking Group"]
MERCHANT_CORPUS = ["McDonald's", "PayPal", "Vodafone", "Argos", "Deliveroo"]


def test_case_and_space_insensitive_match():
    assert suggest_fix("nat west", "bank_name", BANK_CORPUS) == "NatWest"
    assert suggest_fix("metro bank", "bank_name", BANK_CORPUS) == "Metro Bank"


def test_uppercase_punctuation_typo():
    assert suggest_fix("MC DONALDS", "merchant_name", MERCHANT_CORPUS) == "McDonald's"


def test_does_not_suggest_value_for_itself():
    # value already canonical and present in corpus -> no suggestion
    assert suggest_fix("Barclays", "bank_name", BANK_CORPUS) is None


def test_no_match_returns_none():
    assert suggest_fix("Totally Different Co", "bank_name", BANK_CORPUS) is None


def test_none_inputs():
    assert suggest_fix(None, "x", BANK_CORPUS) is None
    assert suggest_fix("x", "x", None) is None


def test_email_domain_typos():
    assert suggest_email_fix("mohammed.taylor@outlok.com") == "mohammed.taylor@outlook.com"
    assert suggest_email_fix("poppy.singh@hotmial.com") == "poppy.singh@hotmail.com"
    assert suggest_email_fix("priya.wilson@icould.com") == "priya.wilson@icloud.com"


def test_email_local_part_preserved():
    fixed = suggest_email_fix("a.b.c+tag@gmial.com")
    assert fixed == "a.b.c+tag@gmail.com"


def test_valid_email_not_changed():
    assert suggest_email_fix("real.user@gmail.com") is None


def test_non_email_returns_none():
    assert suggest_email_fix("not-an-email") is None
    assert suggest_email_fix("a@b@c") is None
