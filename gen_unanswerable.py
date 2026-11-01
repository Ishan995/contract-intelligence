#!/usr/bin/env python
"""Generate 40 realistic unanswerable questions for CUAD contracts."""
import json
import os
from pathlib import Path

from lib_common import split_label

UNANSWERABLE_PATH = "queries/unanswerable.json"


def generate_unanswerable():
    """40 realistic legal questions that CUAD contracts won't answer."""
    questions = [
        "What is the current market capitalization of the parent company?",
        "How many employees does the company currently have on payroll?",
        "What was the company's total revenue in the most recent fiscal year?",
        "What is the CEO's annual base salary and bonus structure?",
        "What environmental compliance certifications does the company hold?",
        "What is the company's current credit rating from Moody's or S&P?",
        "How many pending patent applications does the company have?",
        "What is the company's policy on remote work for employees?",
        "What are the board of directors' meeting minutes from the last quarter?",
        "What is the depreciation schedule for the company's fixed assets?",
        "How many shares of common stock are currently outstanding?",
        "What is the company's diversity and inclusion policy?",
        "What cybersecurity incidents has the company reported in the last year?",
        "What is the company's carbon emissions target for 2030?",
        "How much did the company spend on research and development last year?",
        "What is the company's customer satisfaction score or NPS?",
        "What internal audit findings were reported in the most recent review?",
        "What is the company's policy on political donations and lobbying?",
        "How many data centers does the company operate worldwide?",
        "What is the average tenure of the company's executive leadership team?",
        "What percentage of revenue comes from international markets?",
        "What is the company's workers' compensation claims history?",
        "How many product recalls has the company issued in the past five years?",
        "What is the company's current ratio of debt to equity?",
        "What whistleblower complaints have been filed against the company?",
        "What is the company's GDPR compliance status in European markets?",
        "How many subsidiaries does the parent company currently operate?",
        "What is the company's inventory turnover ratio?",
        "What are the terms of the company's revolving credit facility?",
        "What is the company's policy on stock buybacks?",
        "How many active lawsuits is the company currently defending?",
        "What is the company's estimated total addressable market?",
        "What supply chain disruptions has the company experienced recently?",
        "What is the company's policy on executive stock option vesting?",
        "How many countries does the company have sales operations in?",
        "What is the company's annual marketing and advertising budget?",
        "What safety violations has OSHA cited the company for?",
        "What is the company's free cash flow for the trailing twelve months?",
        "What succession plan exists for the current CEO?",
        "What is the company's stance on cryptocurrency as a payment method?",
    ]

    entries = []
    for q in questions:
        entries.append({
            "query": q,
            "split": split_label(q),
        })

    os.makedirs(os.path.dirname(UNANSWERABLE_PATH), exist_ok=True)
    with open(UNANSWERABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} unanswerable questions to {UNANSWERABLE_PATH}")


if __name__ == "__main__":
    generate_unanswerable()

