from __future__ import annotations

from typing import Dict


class FacultyContextAgent:
    @staticmethod
    def _call(name: str) -> None:
        print(name)

    def ask_for_email(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_email")
        return {"next_action": "ask_email"}

    def ask_for_group_emails(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_group_emails")
        return {"next_action": "ask_group_emails"}

    def ask_for_user_reference_data(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.ask_for_user_reference_data")
        return {"next_action": "ask_user_reference_data"}

    def fetch_faculty_from_source(self) -> Dict[str, str]:
        self._call("FacultyContextAgent.fetch_faculty_from_source")
        return {"next_action": "fetch_faculty_from_source"}


class OpportunityContextAgent:
    @staticmethod
    def _call(name: str) -> None:
        print(name)

    def ask_for_valid_grant_link(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.ask_for_valid_grant_link")
        return {"next_action": "ask_valid_grant_link"}

    def ask_for_grant_identifier(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.ask_for_grant_identifier")
        return {"next_action": "ask_grant_identifier"}

    def ask_for_grant_title_confirmation(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.ask_for_grant_title_confirmation")
        return {"next_action": "ask_grant_title_confirmation"}

    def fetch_grant_from_source(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.fetch_grant_from_source")
        return {"next_action": "fetch_grant_from_source"}

    def search_grant_by_link_in_db(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.search_grant_by_link_in_db")
        return {"next_action": "searched_grant_by_link_in_db"}

    def search_grant_by_title_in_db(self) -> Dict[str, str]:
        self._call("OpportunityContextAgent.search_grant_by_title_in_db")
        return {"next_action": "searched_grant_by_title_in_db"}


class MatchingExecutionAgent:
    @staticmethod
    def _call(name: str) -> None:
        print(name)

    def run_one_to_one_matching(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.run_one_to_one_matching")
        return {"next_action": "return_one_to_one_results"}

    def generate_keywords_for_one_to_one(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.generate_keywords_for_one_to_one")
        return {"next_action": "generated_keywords_one_to_one"}

    def generate_keywords_for_group(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.generate_keywords_for_group")
        return {"next_action": "generated_keywords_group"}

    def generate_keywords_for_group_specific_grant(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.generate_keywords_for_group_specific_grant")
        return {"next_action": "generated_keywords_group_specific_grant"}

    def run_group_matching(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.run_group_matching")
        return {"next_action": "return_group_matching_results"}

    def run_group_matching_with_specific_grant(self) -> Dict[str, str]:
        self._call("MatchingExecutionAgent.run_group_matching_with_specific_grant")
        return {"next_action": "return_group_specific_grant_results"}
