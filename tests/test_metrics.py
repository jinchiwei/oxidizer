"""Tests for oxidizer.scoring.metrics."""
import pytest

from oxidizer.scoring.metrics import (
    ALL_TRANSITIONS,
    compute_active_voice_ratio,
    compute_sentence_lengths,
    compute_transition_score,
    count_banned_words,
    count_contractions,
    count_parentheticals_per_100,
    count_semicolons_per_100,
)


# ---------------------------------------------------------------------------
# compute_sentence_lengths
# ---------------------------------------------------------------------------

class TestComputeSentenceLengths:
    def test_returns_list_of_positive_ints(self):
        text = "The cat sat on the mat. The dog ran away quickly."
        result = compute_sentence_lengths(text)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(n, int) and n > 0 for n in result)

    def test_single_sentence(self):
        text = "This is a single sentence with seven words."
        result = compute_sentence_lengths(text)
        assert len(result) == 1
        assert result[0] == 8

    def test_multiple_sentences(self):
        text = "Hello world. Goodbye cruel world."
        result = compute_sentence_lengths(text)
        assert len(result) == 2
        assert result[0] == 2
        assert result[1] == 3

    def test_empty_string(self):
        assert compute_sentence_lengths("") == []

    def test_whitespace_only(self):
        assert compute_sentence_lengths("   \n\t  ") == []

    def test_longer_academic_text(self):
        text = (
            "We propose a novel registration framework. "
            "The method uses spline-based deformable models. "
            "Results demonstrate improved accuracy over baselines."
        )
        result = compute_sentence_lengths(text)
        assert len(result) == 3
        assert all(n > 0 for n in result)


# ---------------------------------------------------------------------------
# compute_active_voice_ratio
# ---------------------------------------------------------------------------

class TestComputeActiveVoiceRatio:
    def test_empty_string_returns_one(self):
        assert compute_active_voice_ratio("") == 1.0

    def test_active_sentence_returns_high_ratio(self):
        text = "We analyzed the data using regression."
        ratio = compute_active_voice_ratio(text)
        assert 0.0 <= ratio <= 1.0
        assert ratio > 0.5

    def test_passive_sentence_detected(self):
        text = "The data were collected by the researchers."
        ratio = compute_active_voice_ratio(text)
        assert 0.0 <= ratio <= 1.0
        # At least some passive detected means ratio < 1.0
        assert ratio <= 1.0

    def test_ratio_is_in_range(self):
        text = (
            "We analyzed the results. "
            "The patients were recruited from three hospitals. "
            "All measurements were taken by trained staff."
        )
        ratio = compute_active_voice_ratio(text)
        assert 0.0 <= ratio <= 1.0

    def test_fully_active_text(self):
        text = "We ran the experiment. We collected the data. We analyzed the results."
        ratio = compute_active_voice_ratio(text)
        assert ratio == 1.0

    def test_whitespace_only_returns_one(self):
        assert compute_active_voice_ratio("   ") == 1.0


# ---------------------------------------------------------------------------
# count_banned_words
# ---------------------------------------------------------------------------

class TestCountBannedWords:
    BANNED = ["delve", "tapestry", "multifaceted", "it's worth noting", "novel"]

    def test_none_found(self):
        text = "We analyzed the data using regression models."
        result = count_banned_words(text, self.BANNED)
        assert result == []

    def test_single_banned_word_found(self):
        text = "We delve into the details of the approach."
        result = count_banned_words(text, self.BANNED)
        assert "delve" in result
        assert len(result) == 1

    def test_multiple_banned_words_found(self):
        text = "This tapestry of evidence reveals a novel finding."
        result = count_banned_words(text, self.BANNED)
        assert "tapestry" in result
        assert "novel" in result

    def test_case_insensitive_single_word(self):
        text = "The approach is NOVEL and MULTIFACETED."
        result = count_banned_words(text, self.BANNED)
        assert "novel" in result
        assert "multifaceted" in result

    def test_case_insensitive_phrase(self):
        text = "IT'S WORTH NOTING that the results differ."
        result = count_banned_words(text, self.BANNED)
        assert "it's worth noting" in result

    def test_word_boundary_respected(self):
        # "novelty" should NOT match "novel" with word boundaries
        text = "We studied the novelty of the approach."
        result = count_banned_words(text, ["novel"])
        assert result == []

    def test_phrase_found(self):
        text = "It's worth noting that our approach differs."
        result = count_banned_words(text, self.BANNED)
        assert "it's worth noting" in result

    def test_empty_text(self):
        assert count_banned_words("", self.BANNED) == []

    def test_empty_banned_list(self):
        assert count_banned_words("Some text with words.", []) == []

    def test_both_empty(self):
        assert count_banned_words("", []) == []

    def test_each_entry_appears_at_most_once(self):
        text = "We delve and delve again into the delve."
        result = count_banned_words(text, self.BANNED)
        assert result.count("delve") == 1


# ---------------------------------------------------------------------------
# count_semicolons_per_100
# ---------------------------------------------------------------------------

class TestCountSemicolonsPerHundred:
    def test_empty_string(self):
        assert count_semicolons_per_100("") == 0.0

    def test_no_semicolons(self):
        text = "We ran the experiment. Results were significant."
        assert count_semicolons_per_100(text) == 0.0

    def test_single_semicolon_two_sentences(self):
        text = "We ran the experiment; results were significant. This confirms our hypothesis."
        result = count_semicolons_per_100(text)
        # 1 semicolon / 2 sentences * 100 = 50.0
        assert result == pytest.approx(50.0)

    def test_multiple_semicolons(self):
        # Two clear sentences each containing one semicolon → 2 semi / 2 sent * 100 = 100
        text = "We use method A; it is robust. We compare with method B; it is slower."
        result = count_semicolons_per_100(text)
        assert result == pytest.approx(100.0)

    def test_returns_float(self):
        text = "Hello world. Goodbye."
        result = count_semicolons_per_100(text)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# count_parentheticals_per_100
# ---------------------------------------------------------------------------

class TestCountParentheticalsPerHundred:
    def test_empty_string(self):
        assert count_parentheticals_per_100("") == 0.0

    def test_no_parentheses(self):
        text = "We ran the experiment. Results were significant."
        assert count_parentheticals_per_100(text) == 0.0

    def test_single_parenthetical_two_sentences(self):
        text = "We used MRI (magnetic resonance imaging). Results followed."
        result = count_parentheticals_per_100(text)
        # 1 paren / 2 sentences * 100 = 50.0
        assert result == pytest.approx(50.0)

    def test_multiple_parens(self):
        text = "A (note). B (note) (note). C."
        result = count_parentheticals_per_100(text)
        # 3 parens / 3 sentences * 100 = 100.0
        assert result == pytest.approx(100.0)

    def test_returns_float(self):
        text = "Hello world. Goodbye."
        result = count_parentheticals_per_100(text)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# count_contractions
# ---------------------------------------------------------------------------

class TestCountContractions:
    def test_formal_text_zero_contractions(self):
        text = (
            "We analyzed the dataset using multivariate regression. "
            "The results demonstrate significant improvement over the baseline."
        )
        assert count_contractions(text) == 0

    def test_informal_text_has_contractions(self):
        text = "I can't believe it's already over. We don't have enough data."
        result = count_contractions(text)
        assert result >= 2

    def test_common_contractions_detected(self):
        text = "can't won't don't doesn't didn't isn't aren't wasn't weren't"
        result = count_contractions(text)
        assert result >= 8

    def test_we_re_contraction(self):
        text = "We're ready to proceed."
        assert count_contractions(text) >= 1

    def test_lets_contraction(self):
        text = "Let's get started."
        assert count_contractions(text) >= 1

    def test_empty_string(self):
        assert count_contractions("") == 0

    def test_whitespace_only(self):
        assert count_contractions("   ") == 0

    def test_returns_int(self):
        result = count_contractions("I can't do this.")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# compute_transition_score
# ---------------------------------------------------------------------------

class TestComputeTransitionScore:
    PREFERRED = ["however", "additionally", "as such", "as a result", "similarly"]

    def test_empty_string(self):
        assert compute_transition_score("", self.PREFERRED) == 0.0

    def test_whitespace_only(self):
        assert compute_transition_score("   ", self.PREFERRED) == 0.0

    def test_no_transitions(self):
        text = "The cat sat. The dog ran. The bird flew."
        result = compute_transition_score(text, self.PREFERRED)
        assert result == 0.0

    def test_preferred_transitions_score_higher(self):
        preferred_text = "However, we found significant differences. Additionally, the results confirm the hypothesis."
        non_preferred_text = "Furthermore, we found differences. Moreover, the results confirm."
        preferred_score = compute_transition_score(preferred_text, self.PREFERRED)
        non_preferred_score = compute_transition_score(non_preferred_text, self.PREFERRED)
        assert preferred_score > non_preferred_score

    def test_all_preferred_returns_one(self):
        text = "However, this is good. Additionally, we note that similarly the results hold."
        result = compute_transition_score(text, self.PREFERRED)
        # All detected should be in preferred
        assert result == pytest.approx(1.0)

    def test_no_preferred_detected(self):
        text = "Furthermore, we found this. Moreover, the data show that."
        result = compute_transition_score(text, self.PREFERRED)
        # furthermore and moreover are not in PREFERRED
        assert result == 0.0

    def test_returns_float_in_range(self):
        text = "However, we note that furthermore the data confirm the hypothesis."
        result = compute_transition_score(text, self.PREFERRED)
        assert 0.0 <= result <= 1.0

    def test_empty_preferred_list(self):
        text = "However, this is true. Furthermore, we confirm."
        result = compute_transition_score(text, [])
        # Transitions detected but none are preferred → 0.0
        assert result == 0.0


# ---------------------------------------------------------------------------
# ALL_TRANSITIONS sanity checks
# ---------------------------------------------------------------------------

class TestAllTransitions:
    def test_is_non_empty_list(self):
        assert isinstance(ALL_TRANSITIONS, list)
        assert len(ALL_TRANSITIONS) > 0

    def test_all_strings(self):
        assert all(isinstance(t, str) for t in ALL_TRANSITIONS)

    def test_common_transitions_present(self):
        lower = [t.lower() for t in ALL_TRANSITIONS]
        for expected in ["however", "therefore", "furthermore", "additionally"]:
            assert expected in lower
