from vlmc.vqa_metric import clean_prediction, normalize, vqa_accuracy


def test_exact_and_normalized():
    assert vqa_accuracy("2", ["2"] * 10) == 1.0
    assert vqa_accuracy("two", ["2"] * 10) == 1.0          # number words
    assert vqa_accuracy("A dog.", ["dog"] * 10) == 1.0     # article + period
    assert vqa_accuracy("cat", ["dog"] * 10) == 0.0


def test_leave_one_out():
    # 3/10 humans agree: 3 subsets see 2 matches (2/3), 7 subsets see 3 matches (1.0) -> 0.9
    gts = ["red"] * 3 + ["blue"] * 7
    assert abs(vqa_accuracy("red", gts) - 0.9) < 1e-9
    # 1/10 agree: 9 subsets see 1 match (1/3), 1 subset sees 0 -> 0.3
    assert abs(vqa_accuracy("red", ["red"] + ["blue"] * 9) - 0.3) < 1e-9


def test_contractions_and_cleaning():
    assert normalize("dont") == "don't"
    assert clean_prediction("Yes.\nBecause...") == "Yes"
    assert clean_prediction("Answer: 3") == "3"
