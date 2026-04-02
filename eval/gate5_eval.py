#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

import base64
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SSH_TARGET = "studio2@100.93.190.120"
API_URL = "http://127.0.0.1:52415/v1/chat/completions"
MODEL = "mlx-community/Kimi-K2.5"
OUT_PATH = Path("/Users/tmuxbot/exo-fork/eval/results/gate5_phase2.json")


@dataclass
class TestCase:
    case_id: str
    category: str
    image_url: str
    question: str
    expected: Any
    scorer: str


def svg_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def build_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    ocr_texts = [
        "HELLO 123",
        "EXO VISION TEST",
        "GATE FIVE",
        "KIMI K2.5",
        "PASS RATE 60",
    ]
    for idx, txt in enumerate(ocr_texts, 1):
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='480' height='160'>
<rect width='100%' height='100%' fill='white'/>
<text x='20' y='95' font-size='52' font-family='Arial' fill='black'>{txt}</text>
</svg>"""
        cases.append(
            TestCase(
                case_id=f"ocr_{idx}",
                category="OCR",
                image_url=svg_data_url(svg),
                question="Extract the exact visible text from this image.",
                expected=txt,
                scorer="ocr",
            )
        )

    spatial_specs = [
        ("A", "B", "above", "Is A above or below B?"),
        ("LEFT", "RIGHT", "left", "Is LEFT to the left or right of RIGHT?"),
        ("TOP", "BOTTOM", "above", "Is TOP above or below BOTTOM?"),
        ("SUN", "GRASS", "above", "Is SUN above or below GRASS?"),
        ("CAT", "DOG", "left", "Is CAT left or right of DOG?"),
    ]
    for idx, (first, second, expected, question) in enumerate(spatial_specs, 1):
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='520' height='220'>
<rect width='100%' height='100%' fill='white'/>
<text x='40' y='70' font-size='42' font-family='Arial' fill='black'>{first}</text>
<text x='320' y='180' font-size='42' font-family='Arial' fill='black'>{second}</text>
</svg>"""
        cases.append(
            TestCase(
                case_id=f"spatial_{idx}",
                category="Spatial reasoning",
                image_url=svg_data_url(svg),
                question=question,
                expected=expected,
                scorer="spatial",
            )
        )

    count_values = [2, 4, 6, 8, 10]
    for idx, count in enumerate(count_values, 1):
        circles = []
        for i in range(count):
            x = 45 + (i % 5) * 90
            y = 55 + (i // 5) * 95
            circles.append(f"<circle cx='{x}' cy='{y}' r='28' fill='red'/>")
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='220'>"
            "<rect width='100%' height='100%' fill='white'/>"
            + "".join(circles)
            + "</svg>"
        )
        cases.append(
            TestCase(
                case_id=f"count_{idx}",
                category="Counting",
                image_url=svg_data_url(svg),
                question="How many red circles are in this image? Reply with a number.",
                expected=count,
                scorer="counting",
            )
        )

    desc_svgs = [
        (
            "desc_1",
            """<svg xmlns='http://www.w3.org/2000/svg' width='520' height='260'>
<rect width='100%' height='100%' fill='#d9f0ff'/>
<circle cx='430' cy='55' r='32' fill='yellow'/>
<rect x='70' y='150' width='150' height='90' fill='#dd8c5a'/>
<polygon points='60,150 145,95 230,150' fill='#b24a2f'/>
<rect x='320' y='130' width='28' height='110' fill='#6b4b2a'/>
<circle cx='334' cy='118' r='38' fill='#2f9b4d'/>
</svg>""",
            ["house", "sun", "tree"],
        ),
        (
            "desc_2",
            """<svg xmlns='http://www.w3.org/2000/svg' width='520' height='260'>
<rect width='100%' height='100%' fill='white'/>
<circle cx='95' cy='120' r='42' fill='#f58aa9'/>
<circle cx='210' cy='120' r='42' fill='#8ac6f5'/>
<circle cx='325' cy='120' r='42' fill='#8ef58a'/>
<text x='55' y='220' font-size='36' font-family='Arial'>balloons</text>
</svg>""",
            ["balloon", "three", "color"],
        ),
        (
            "desc_3",
            """<svg xmlns='http://www.w3.org/2000/svg' width='520' height='260'>
<rect width='100%' height='100%' fill='#e8f0ff'/>
<rect x='0' y='190' width='520' height='70' fill='#8fd38f'/>
<path d='M40 190 Q120 115 200 190' stroke='#4f6db1' stroke-width='24' fill='none'/>
<path d='M200 190 Q280 115 360 190' stroke='#4f6db1' stroke-width='24' fill='none'/>
<text x='360' y='90' font-size='34' font-family='Arial'>cloud</text>
</svg>""",
            ["bridge", "river", "cloud"],
        ),
        (
            "desc_4",
            """<svg xmlns='http://www.w3.org/2000/svg' width='520' height='260'>
<rect width='100%' height='100%' fill='#fdf7e3'/>
<rect x='60' y='70' width='400' height='130' rx='12' fill='white' stroke='black'/>
<text x='95' y='150' font-size='54' font-family='Arial' fill='black'>COFFEE</text>
<circle cx='420' cy='135' r='28' fill='#6b4b2a'/>
</svg>""",
            ["coffee", "sign", "text"],
        ),
        (
            "desc_5",
            "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png",
            ["python", "logo", "blue"],
        ),
    ]

    for case_id, image_url_or_svg, keywords in desc_svgs:
        image_url = (
            image_url_or_svg
            if image_url_or_svg.startswith("http")
            else svg_data_url(image_url_or_svg)
        )
        cases.append(
            TestCase(
                case_id=case_id,
                category="Visual description",
                image_url=image_url,
                question="Describe what is visible in this image in one short sentence.",
                expected=keywords,
                scorer="description",
            )
        )

    assert len(cases) == 20
    return cases


def run_remote_request(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    payload_json = json.dumps(payload)
    remote = f"""python3 - <<'PY'
import json
import urllib.error
import urllib.request

url = {API_URL!r}
payload = json.loads({payload_json!r})
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode('utf-8'),
    headers={{'Content-Type': 'application/json'}},
    method='POST',
)
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        print(resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    body = e.read().decode('utf-8', errors='ignore')
    print(json.dumps({{'error': f'HTTP {{e.code}}', 'body': body}}))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
PY"""
    proc = subprocess.run(
        ["ssh", SSH_TARGET, remote],
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return None, f"ssh failed: {proc.stderr.strip() or proc.stdout.strip()}"
    raw = proc.stdout.strip()
    if not raw:
        return None, "empty response"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid json response: {e}"
    if isinstance(parsed, dict) and "error" in parsed:
        return parsed, str(parsed.get("error"))
    return parsed, None


def extract_answer(resp: dict[str, Any]) -> str:
    try:
        msg = resp["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return " ".join(parts).strip()
    except Exception:
        pass
    return ""


def score_ocr(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def word_to_num(token: str) -> int | None:
    mapping = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    return mapping.get(token.lower())


def extract_number(answer: str) -> int | None:
    digit = re.search(r"-?\d+", answer)
    if digit:
        return int(digit.group(0))
    for token in re.findall(r"[a-zA-Z]+", answer.lower()):
        n = word_to_num(token)
        if n is not None:
            return n
    return None


def score_counting(answer: str, expected: int) -> tuple[bool, int | None]:
    got = extract_number(answer)
    if got is None:
        return False, None
    return abs(got - expected) <= 1, got


def score_spatial(answer: str, expected_term: str) -> bool:
    return expected_term.lower() in answer.lower()


def score_description(
    answer: str, expected_keywords: list[str]
) -> tuple[bool, list[str]]:
    text = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in text]
    return len(found) >= 2, found


def evaluate_case(case: TestCase) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case.question},
                    {"type": "image_url", "image_url": {"url": case.image_url}},
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0,
    }

    response, err = run_remote_request(payload)
    if response is None:
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected": case.expected,
            "answer": "",
            "status": "SKIPPED",
            "score": 0,
            "error": err,
        }

    if isinstance(response, dict) and "error" in response:
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected": case.expected,
            "answer": "",
            "status": "SKIPPED",
            "score": 0,
            "error": response.get("error"),
            "error_body": response.get("body", ""),
        }

    answer = extract_answer(response)
    passed = False
    details: dict[str, Any] = {}
    if case.scorer == "ocr":
        passed = score_ocr(answer, str(case.expected))
    elif case.scorer == "counting":
        passed, extracted = score_counting(answer, int(case.expected))
        details["extracted_number"] = extracted
    elif case.scorer == "spatial":
        passed = score_spatial(answer, str(case.expected))
    elif case.scorer == "description":
        passed, found = score_description(answer, list(case.expected))
        details["matched_keywords"] = found

    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected": case.expected,
        "answer": answer,
        "status": "PASS" if passed else "FAIL",
        "score": 1 if passed else 0,
        "details": details,
    }


def summarize(
    results: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], int, int]:
    categories = ["OCR", "Spatial reasoning", "Counting", "Visual description"]
    summary = {c: {"passed": 0, "total": 0, "skipped": 0} for c in categories}
    for r in results:
        cat = r["category"]
        if r["status"] == "SKIPPED":
            summary[cat]["skipped"] += 1
            continue
        summary[cat]["total"] += 1
        summary[cat]["passed"] += int(r["score"])

    overall_total = sum(v["total"] for v in summary.values())
    overall_passed = sum(v["passed"] for v in summary.values())
    return summary, overall_passed, overall_total


def print_summary(
    summary: dict[str, dict[str, int]], overall_passed: int, overall_total: int
) -> None:
    print("\nGate 5 Phase 2 Summary")
    print("category | passed | total | pass_rate")
    print("-" * 44)
    for cat in ["OCR", "Spatial reasoning", "Counting", "Visual description"]:
        passed = summary[cat]["passed"]
        total = summary[cat]["total"]
        skipped = summary[cat]["skipped"]
        rate = (passed / total * 100.0) if total else 0.0
        skip_suffix = f" (skipped={skipped})" if skipped else ""
        print(f"{cat} | {passed} | {total} | {rate:.1f}%{skip_suffix}")
    overall_rate = (overall_passed / overall_total * 100.0) if overall_total else 0.0
    required_passes = 12
    overall_ok = overall_passed >= required_passes
    print("-" * 44)
    print(f"overall | {overall_passed} | {overall_total} | {overall_rate:.1f}%")
    print(f"overall_criterion (>=12 passes): {'PASS' if overall_ok else 'FAIL'}")


def main() -> int:
    cases = build_cases()
    results: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        print(f"[{idx:02d}/{len(cases)}] {case.case_id} ({case.category})")
        result = evaluate_case(case)
        results.append(result)
        print(f"  -> {result['status']}")

    summary, overall_passed, overall_total = summarize(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model": MODEL,
        "api_url": API_URL,
        "overall": {
            "passed": overall_passed,
            "total": overall_total,
            "criterion_pass": overall_passed >= 12,
        },
        "summary": summary,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print_summary(summary, overall_passed, overall_total)
    print(f"\nWrote results to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
