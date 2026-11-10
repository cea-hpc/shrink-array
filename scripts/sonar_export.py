#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Every SonarQube Cloud issue of a branch, in one JSON or CSV file.

The analysis is .github/workflows/sonar.yml's (project cea-hpc_shrink-array);
this fetches its issues through the Web API (api/issues/search), page after
page, with the rule names, and writes them sorted by file and line. A summary
by severity and by rule goes to stderr.

Usage: scripts/sonar_export.py [--branch develop] [--format json|csv]
                               [--all] [--hotspots] [-o FILE] [--project KEY]
  --branch    the analysed branch (default: develop; main is the main branch)
  --format    json (default) or csv; -o's suffix chooses it when given
  --all       the resolved issues too (default: the open ones only)
  --hotspots  the security hotspots too (api/hotspots/search), in the JSON
              under "hotspots" (CSV: as rows of type SECURITY_HOTSPOT)
  -o FILE     where to write (default: stdout)
  --project   the project key (default: $SONAR_PROJECT_KEY, else
              cea-hpc_shrink-array)

Environment:
  SONAR_TOKEN  a user token (sonarcloud.io, My Account, Security). The project
               is public: without a token the issues are still readable.
               Given to curl on its standard input, never on its command line.
  SONAR_HOST   default https://sonarcloud.io
  https_proxy  the shell's proxy: curl goes through it with Kerberos
               (--proxy-negotiate: the CEA proxy accepts nothing else), as
               scripts/coverity_submit.sh does.

The API returns at most 10,000 issues per query: past that, the query is
split by language, type, severity, rule or directory (SPLIT_FACETS). Python 3.6 compatible (the login nodes').
"""

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import urllib.parse

DEFAULT_PROJECT = os.environ.get("SONAR_PROJECT_KEY") or "cea-hpc_shrink-array"
PAGE_SIZE = 500
API_LIMIT = 10000

CSV_FIELDS = [
    "file", "line", "type", "severity", "impacts", "rule", "rule_name", "message",
    "status", "resolution", "effort", "tags", "creation_date", "key",
]


def die(msg):
    sys.stderr.write("sonar_export: %s\n" % msg)
    sys.exit(1)


def get(host, path, params):
    """GET host/path?params as JSON, through curl (proxy, Kerberos, token)."""
    url = "%s%s?%s" % (host, path, urllib.parse.urlencode(params))
    # The token goes through curl's config on stdin, not argv (ps shows argv).
    config = ""
    token = os.environ.get("SONAR_TOKEN", "")
    if token:
        config = 'header = "Authorization: Bearer %s"\n' % token
    # The HTTP status on a last line of its own (-w): --fail-with-body needs
    # curl 7.76, the login nodes have 7.61.
    cmd = ["curl", "-sS", "--max-time", "120", "--proxy-negotiate", "-U", ":", "-w", "\n%{http_code}",
           "-K", "-", url]
    proc = subprocess.run(cmd, input=config.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    body, _, status = proc.stdout.decode().rpartition("\n")
    if proc.returncode != 0:
        die("%s: curl exit %d: %s" % (path, proc.returncode, proc.stderr.decode().strip()))
    if status != "200":
        die("%s: HTTP %s: %s" % (path, status, body[:300]))
    try:
        return json.loads(body)
    except ValueError:
        die("%s: not JSON: %s" % (path, body[:300]))


def search(host, base, extra=None):
    """Every issue of one query, page after page; also the rule names it saw."""
    params = dict(base)
    params.update(extra or {})
    params.update({"ps": PAGE_SIZE, "additionalFields": "rules"})
    issues, rules = [], {}
    page = 1
    while True:
        params["p"] = page
        data = get(host, "/api/issues/search", params)
        issues.extend(data.get("issues", []))
        for r in data.get("rules", []):
            rules[r["key"]] = r.get("name", "")
        total = data.get("paging", {}).get("total", data.get("total", 0))
        if page * PAGE_SIZE >= min(total, API_LIMIT):
            return issues, rules, total
        page += 1


# Facets that partition the issues (each issue has exactly one value of
# each): a query over the API's limit is split along the first one whose
# values cover it -- a facet lists at most 100 values, so rules or
# directories may not.
SPLIT_FACETS = ["languages", "types", "severities", "rules", "directories"]


def all_issues(host, base, facets=None):
    """The issues of one query, split along a facet while it holds too many."""
    facets = SPLIT_FACETS if facets is None else facets
    first = get(host, "/api/issues/search", dict(base, ps=1, facets=",".join(facets) or "rules"))
    total = first.get("paging", {}).get("total", first.get("total", 0))
    if total <= API_LIMIT:
        issues, rules, _ = search(host, base)
        return issues, rules
    by_facet = {f.get("property"): f.get("values", []) for f in first.get("facets", [])}
    for name in facets:
        values = [v for v in by_facet.get(name, []) if v.get("count")]
        if len(values) < 2 or sum(v["count"] for v in values) != total:
            continue
        issues, rules = [], {}
        rest = [f for f in facets if f != name]
        for v in values:
            part, names = all_issues(host, dict(base, **{name: v["val"]}), rest)
            issues.extend(part)
            rules.update(names)
        return issues, rules
    sys.stderr.write("sonar_export: %d issues for %s: no facet splits them under %d, the first %d kept\n"
                     % (total, {k: v for k, v in base.items() if k in SPLIT_FACETS}, API_LIMIT, API_LIMIT))
    issues, rules, _ = search(host, base)
    return issues, rules


def all_hotspots(host, project, branch):
    params = {"projectKey": project, "branch": branch, "ps": PAGE_SIZE}
    hotspots, page = [], 1
    while True:
        params["p"] = page
        data = get(host, "/api/hotspots/search", params)
        hotspots.extend(data.get("hotspots", []))
        if page * PAGE_SIZE >= min(data.get("paging", {}).get("total", 0), API_LIMIT):
            return hotspots
        page += 1


def file_of(component):
    """cea-hpc_shrink-array:chunk/x.h -> chunk/x.h"""
    return component.split(":", 1)[1] if ":" in component else component


def row(issue, rules):
    impacts = issue.get("impacts", [])
    return {
        "file": file_of(issue.get("component", "")),
        "line": issue.get("line", ""),
        "type": issue.get("type", ""),
        "severity": issue.get("severity", ""),
        "impacts": " ".join("%s:%s" % (i.get("softwareQuality"), i.get("severity")) for i in impacts),
        "rule": issue.get("rule", ""),
        "rule_name": rules.get(issue.get("rule", ""), ""),
        "message": issue.get("message", ""),
        "status": issue.get("issueStatus", issue.get("status", "")),
        "resolution": issue.get("resolution", ""),
        "effort": issue.get("effort", ""),
        "tags": " ".join(issue.get("tags", [])),
        "creation_date": issue.get("creationDate", ""),
        "key": issue.get("key", ""),
    }


def hotspot_row(h):
    return {
        "file": file_of(h.get("component", "")),
        "line": h.get("line", ""),
        "type": "SECURITY_HOTSPOT",
        "severity": h.get("vulnerabilityProbability", ""),
        "impacts": "",
        "rule": h.get("ruleKey", ""),
        "rule_name": h.get("securityCategory", ""),
        "message": h.get("message", ""),
        "status": h.get("status", ""),
        "resolution": h.get("resolution", ""),
        "effort": "",
        "tags": "",
        "creation_date": h.get("creationDate", ""),
        "key": h.get("key", ""),
    }


def sort_key(r):
    line = r["line"] if isinstance(r["line"], int) else 0
    return (r["file"], line, r["rule"])


def summary(rows, total_label):
    sys.stderr.write("%d %s\n" % (len(rows), total_label))
    by_sev = collections.Counter(r["severity"] or "?" for r in rows)
    sys.stderr.write("  by severity: %s\n" % ", ".join("%s %d" % kv for kv in by_sev.most_common()))
    by_rule = collections.Counter((r["rule"], r["rule_name"]) for r in rows)
    sys.stderr.write("  most frequent rules:\n")
    for (rule, name), n in by_rule.most_common(15):
        sys.stderr.write("    %5d  %-14s %s\n" % (n, rule, name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--branch", default="develop")
    ap.add_argument("--format", choices=["json", "csv"])
    ap.add_argument("--all", action="store_true", help="the resolved issues too")
    ap.add_argument("--hotspots", action="store_true", help="the security hotspots too")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    fmt = args.format
    if fmt is None:
        fmt = "csv" if args.output and args.output.endswith(".csv") else "json"
    host = os.environ.get("SONAR_HOST", "https://sonarcloud.io").rstrip("/")

    base = {"componentKeys": args.project, "branch": args.branch}
    if not args.all:
        base["resolved"] = "false"
    issues, rules = all_issues(host, base)
    # Once each, whatever the split (a directory filter may take its
    # subdirectories along).
    issues = list({i["key"]: i for i in issues}.values())
    rows = sorted((row(i, rules) for i in issues), key=sort_key)
    hotspots = all_hotspots(host, args.project, args.branch) if args.hotspots else []
    hrows = sorted((hotspot_row(h) for h in hotspots), key=sort_key)
    summary(rows, "issues (%s, branch %s)" % ("all" if args.all else "open", args.branch))
    if args.hotspots:
        sys.stderr.write("%d security hotspots\n" % len(hrows))

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        if fmt == "json":
            doc = {"project": args.project, "branch": args.branch, "host": host, "issues": rows}
            if args.hotspots:
                doc["hotspots"] = hrows
            json.dump(doc, out, indent=1, ensure_ascii=False)
            out.write("\n")
        else:
            w = csv.DictWriter(out, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows + hrows)
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
