"""Tests for the FAQ knowledge base tool."""

from __future__ import annotations

from src.tools.faq import _split_into_sections, get_full_faq, search_faq


class TestFaqLoading:
    def test_faq_loads_successfully(self):
        faq = get_full_faq()
        assert len(faq) > 0
        assert "Acme Dental" in faq

    def test_faq_contains_key_sections(self):
        faq = get_full_faq()
        assert "Services & Appointments" in faq
        assert "Booking, Rescheduling & Cancellation" in faq
        assert "Pricing & Payment" in faq

    def test_split_into_sections_returns_qa_pairs(self):
        faq = get_full_faq()
        sections = _split_into_sections(faq)
        assert len(sections) > 10  # We know there are many Q&A pairs

        # Check that each section has a heading and body
        for section in sections:
            assert "heading" in section
            assert "body" in section
            assert len(section["heading"]) > 0


class TestSearchFaq:
    def test_search_pricing_returns_cost_info(self):
        result = search_faq.invoke({"query": "how much does a check-up cost"})
        assert "60" in result or "€60" in result

    def test_search_cancellation_returns_policy(self):
        result = search_faq.invoke({"query": "cancellation policy"})
        assert "24 hour" in result.lower() or "cancel" in result.lower()

    def test_search_services_returns_checkup_info(self):
        result = search_faq.invoke({"query": "what services do you offer"})
        assert "check-up" in result.lower() or "dental" in result.lower()

    def test_search_insurance_returns_info(self):
        result = search_faq.invoke({"query": "do you accept insurance"})
        assert "insurance" in result.lower() or "receipt" in result.lower()

    def test_search_no_match_returns_summary(self):
        result = search_faq.invoke({"query": "xyz123 completely unrelated topic"})
        assert "Acme Dental" in result or "couldn't find" in result.lower()

    def test_search_student_discount(self):
        result = search_faq.invoke({"query": "student discount price"})
        assert "50" in result or "student" in result.lower()

    def test_search_walk_ins(self):
        result = search_faq.invoke({"query": "walk in appointment"})
        assert "walk" in result.lower() or "book" in result.lower()
