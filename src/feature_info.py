"""Plain-English feature metadata, transcribed from the official UCI feature document
(docs/source/uci_features_description.txt). Used by reports/features.md and app.py.

``rule`` paraphrases the document's extraction rule. ``values`` gives the conventional
reading of each code (-1 = phishing-like, 0 = suspicious, 1 = legitimate-like). The UCI
documentation never states the numeric codes, so these readings are a convention,
supported but not proven by the Phase 1 checks. ``note`` records known ambiguities found
in Phases 1-2.
"""

from __future__ import annotations

PHISHY, SUSP, LEGIT = "phishing-like", "suspicious", "legitimate-like"

FEATURES: dict[str, dict] = {
    "having_ip_address": {
        "label": "IP address used as the domain",
        "group": "URL / address bar",
        "rule": "Domain part of the URL is an IP address (e.g. http://125.98.3.123/...) → phishing.",
        "values": {-1: "IP address in URL (" + PHISHY + ")", 1: "normal domain name"},
    },
    "url_length": {
        "label": "URL length",
        "group": "URL / address bar",
        "rule": "Length < 54 → legitimate; 54–75 → suspicious; > 75 → phishing.",
        "values": {-1: "long URL (> 75 chars)", 0: "medium (54–75 chars)", 1: "short (< 54 chars)"},
        "note": "Level 0 is rare after cleaning (96 rows).",
    },
    "shortining_service": {
        "label": "URL shortening service (e.g. bit.ly)",
        "group": "URL / address bar",
        "rule": "URL uses a shortening service such as TinyURL → phishing.",
        "values": {-1: "shortened URL (" + PHISHY + ")", 1: "not shortened"},
        "note": "Spelling 'shortining' is from the source. Observed association with the label is weak and slightly inverted.",
    },
    "having_at_symbol": {
        "label": "'@' symbol in URL",
        "group": "URL / address bar",
        "rule": "URL contains '@' (browser ignores everything before it) → phishing.",
        "values": {-1: "contains '@'", 1: "no '@'"},
    },
    "double_slash_redirecting": {
        "label": "'//' redirect inside the URL path",
        "group": "URL / address bar",
        "rule": "Last occurrence of '//' is beyond position 7 → phishing.",
        "values": {-1: "'//' inside path", 1: "no extra '//'"},
    },
    "prefix_suffix": {
        "label": "Dash '-' in the domain name",
        "group": "URL / address bar",
        "rule": "Domain name contains '-' (e.g. confirme-paypal.com) → phishing.",
        "values": {-1: "dash present (" + PHISHY + ")", 1: "no dash"},
        "note": "All 4,898 raw phishing rows have -1, yet -1 is the majority code (87% of rows). "
                "This contradicts the document's description of '-' as rare; possible coding or collection artifact.",
    },
    "having_sub_domain": {
        "label": "Number of sub-domains",
        "group": "URL / address bar",
        "rule": "Dots in domain (after removing www. and ccTLD): 1 → legitimate; 2 → suspicious; more → phishing.",
        "values": {-1: "multiple sub-domains", 0: "one sub-domain", 1: "no sub-domain"},
        "note": "Non-monotone: code 0 has a higher phishing rate than -1.",
    },
    "sslfinal_state": {
        "label": "HTTPS certificate status",
        "group": "URL / address bar",
        "rule": "HTTPS with trusted issuer and certificate age ≥ 1 year → legitimate; HTTPS with untrusted issuer → "
                "suspicious; otherwise → phishing.",
        "values": {-1: "no HTTPS / invalid", 0: "HTTPS, untrusted issuer", 1: "HTTPS, trusted & ≥ 1 year"},
        "note": "Strongest single predictor. Non-monotone: code 0 (99% phishing) is riskier than -1. The document text "
                "mentions both 1-year and 2-year certificate ages.",
    },
    "domain_registration_length": {
        "label": "Domain registration length",
        "group": "URL / address bar",
        "rule": "Domain expires in ≤ 1 year → phishing.",
        "values": {-1: "expires within 1 year", 1: "registered for longer"},
        "note": "Observed association is the reverse of the rule: -1 is associated with LEGITIMATE sites "
                "(43% vs 67% phishing). Cannot be resolved from the documentation.",
    },
    "favicon": {
        "label": "Favicon loaded from external domain",
        "group": "URL / address bar",
        "rule": "Favicon loaded from a different domain than the address bar → phishing.",
        "values": {-1: "external favicon", 1: "same-domain favicon"},
        "note": "Strongly co-occurs with popupwindow (Cramér's V 0.93) and port; near-zero label association.",
    },
    "port": {
        "label": "Non-standard port status",
        "group": "URL / address bar",
        "rule": "Server ports differ from their preferred open/closed status → phishing.",
        "values": {-1: "non-preferred port status", 1: "preferred port status"},
        "note": "The document's rule text is self-contradictory ('of the preferred status → phishing').",
    },
    "https_token": {
        "label": "'https' token in the domain part",
        "group": "URL / address bar",
        "rule": "Domain part contains the text 'https' (e.g. https-www-paypal...) → phishing.",
        "values": {-1: "'https' in domain text", 1: "no token"},
    },
    "request_url": {
        "label": "External objects (images, media)",
        "group": "Abnormal / page content",
        "rule": "Share of embedded objects loaded from other domains: < 22% → legitimate; otherwise phishing "
                "(document also mentions a 22–61% suspicious band).",
        "values": {-1: "many external objects", 1: "mostly same-domain objects"},
    },
    "url_of_anchor": {
        "label": "Anchor (<a>) links to other domains or nowhere",
        "group": "Abnormal / page content",
        "rule": "% of anchors pointing elsewhere or to '#'/javascript: < 31% → legitimate; 31–67% → suspicious; "
                "> 67% → phishing.",
        "values": {-1: "> 67% (" + PHISHY + ")", 0: "31–67%", 1: "< 31%"},
    },
    "links_in_tags": {
        "label": "Links in <meta>, <script>, <link> tags",
        "group": "Abnormal / page content",
        "rule": "% of these links to other domains: < 17% → legitimate; 17–81% → suspicious; > 81% → phishing.",
        "values": {-1: "> 81%", 0: "17–81%", 1: "< 17%"},
    },
    "sfh": {
        "label": "Server Form Handler (form action)",
        "group": "Abnormal / page content",
        "rule": "Form action empty or 'about:blank' → phishing; points to a different domain → suspicious; "
                "otherwise legitimate.",
        "values": {-1: "empty / about:blank", 0: "different domain", 1: "same domain"},
    },
    "submitting_to_email": {
        "label": "Form submits to e-mail",
        "group": "Abnormal / page content",
        "rule": "Uses mail() or mailto: to submit user information → phishing.",
        "values": {-1: "submits to e-mail", 1: "does not"},
    },
    "abnormal_url": {
        "label": "Host name missing from URL (WHOIS)",
        "group": "Abnormal / page content",
        "rule": "Host name from WHOIS not included in the URL → phishing.",
        "values": {-1: "identity not in URL", 1: "identity in URL"},
    },
    "redirect": {
        "label": "Number of redirects",
        "group": "HTML / JavaScript",
        "rule": "Redirects ≤ 1 → legitimate; 2–3 → suspicious; ≥ 4 → phishing.",
        "values": {0: "code 0 (meaning undocumented)", 1: "code 1 (meaning undocumented)"},
        "note": "Coded {0, 1}, not {-1, 1}; which redirect band each code represents is undocumented.",
    },
    "on_mouseover": {
        "label": "onMouseOver changes the status bar",
        "group": "HTML / JavaScript",
        "rule": "JavaScript onMouseOver alters the status bar (fake link display) → phishing.",
        "values": {-1: "status bar altered", 1: "not altered"},
    },
    "rightclick": {
        "label": "Right-click disabled",
        "group": "HTML / JavaScript",
        "rule": "Page disables right-click (event.button==2) → phishing.",
        "values": {-1: "right-click disabled", 1: "enabled"},
    },
    "popupwindow": {
        "label": "Pop-up window with text fields",
        "group": "HTML / JavaScript",
        "rule": "Pop-up window asks for information in text fields → phishing.",
        "values": {-1: "pop-up with text fields", 1: "no such pop-up"},
    },
    "iframe": {
        "label": "Invisible iframe",
        "group": "HTML / JavaScript",
        "rule": "Page uses an iframe (typically without border) → phishing.",
        "values": {-1: "uses iframe", 1: "no iframe"},
    },
    "age_of_domain": {
        "label": "Age of domain",
        "group": "Domain / reputation",
        "rule": "Domain age ≥ 6 months (WHOIS) → legitimate; otherwise phishing.",
        "values": {-1: "younger than 6 months", 1: "6 months or older"},
    },
    "dnsrecord": {
        "label": "DNS record exists",
        "group": "Domain / reputation",
        "rule": "No DNS/WHOIS record for the domain → phishing.",
        "values": {-1: "no DNS record", 1: "DNS record found"},
    },
    "web_traffic": {
        "label": "Website traffic rank (Alexa)",
        "group": "Domain / reputation",
        "rule": "Alexa rank < 100,000 → legitimate; > 100,000 → suspicious; no rank → phishing.",
        "values": {-1: "no traffic rank", 0: "rank > 100,000", 1: "rank < 100,000"},
        "note": "Non-monotone: code 0 has a higher phishing rate than -1. Alexa rankings were retired in 2022.",
    },
    "page_rank": {
        "label": "Google PageRank",
        "group": "Domain / reputation",
        "rule": "PageRank < 0.2 → phishing.",
        "values": {-1: "PageRank < 0.2", 1: "PageRank ≥ 0.2"},
        "note": "Public PageRank values are no longer published.",
    },
    "google_index": {
        "label": "Indexed by Google",
        "group": "Domain / reputation",
        "rule": "Page not in Google's index → phishing.",
        "values": {-1: "not indexed", 1: "indexed"},
    },
    "links_pointing_to_page": {
        "label": "External links pointing to the page",
        "group": "Domain / reputation",
        "rule": "0 links → phishing; 1–2 → suspicious; more → legitimate.",
        "values": {-1: "no links", 0: "1–2 links", 1: "more than 2 links"},
    },
    "statistical_report": {
        "label": "Listed in phishing statistics reports",
        "group": "Domain / reputation",
        "rule": "Host in PhishTank top-10 domains/IPs or StopBadware top-50 IPs (2010–2012) → phishing.",
        "values": {-1: "listed", 1: "not listed"},
    },
}
