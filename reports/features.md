# Feature descriptions — UCI Phishing Websites

Source: the feature document distributed with the dataset (`docs/source/Phishing Websites Features.docx`, text in `docs/source/uci_features_description.txt`), Mohammad, Thabtah & McCluskey; dataset licensed CC BY 4.0.

**How to read the codes.** The documentation describes each indicator as *phishing / suspicious / legitimate* but never states the numeric codes. This project uses the conventional reading −1 = phishing-like, 0 = suspicious, 1 = legitimate-like, which the Phase 1 checks support (9/9 anomaly features have −1 as the rare code) but do not prove. Observed statistics below come from executed code: counts use all 5,721 cleaned rows; phishing rates and Cramér's V use the training partition only; ROC-AUC is a one-feature model fitted on train and scored on validation. All are associations, not causal effects.

## Summary

| Feature | Group | Levels | Cramér's V (train) | 1-feature val ROC-AUC |
|---|---|---|---|---|
| `having_ip_address` | URL / address bar | -1, 1 | 0.095 | 0.554 |
| `url_length` | URL / address bar | -1, 0, 1 | 0.071 | 0.545 |
| `shortining_service` | URL / address bar | -1, 1 | 0.072 | 0.542 |
| `having_at_symbol` | URL / address bar | -1, 1 | 0.045 | 0.503 |
| `double_slash_redirecting` | URL / address bar | -1, 1 | 0.025 | 0.529 |
| `prefix_suffix` | URL / address bar | -1, 1 | 0.384 | 0.639 |
| `having_sub_domain` | URL / address bar | -1, 0, 1 | 0.400 | 0.692 |
| `sslfinal_state` | URL / address bar | -1, 0, 1 | 0.778 | 0.898 |
| `domain_registration_length` | URL / address bar | -1, 1 | 0.220 | 0.599 |
| `favicon` | URL / address bar | -1, 1 | 0.005 | 0.530 |
| `port` | URL / address bar | -1, 1 | 0.018 | 0.488 |
| `https_token` | URL / address bar | -1, 1 | 0.042 | 0.530 |
| `request_url` | Abnormal / page content | -1, 1 | 0.237 | 0.639 |
| `url_of_anchor` | Abnormal / page content | -1, 0, 1 | 0.718 | 0.858 |
| `links_in_tags` | Abnormal / page content | -1, 0, 1 | 0.254 | 0.655 |
| `sfh` | Abnormal / page content | -1, 0, 1 | 0.283 | 0.634 |
| `submitting_to_email` | Abnormal / page content | -1, 1 | 0.000 | 0.483 |
| `abnormal_url` | Abnormal / page content | -1, 1 | 0.086 | 0.537 |
| `redirect` | HTML / JavaScript | 0, 1 | 0.043 | 0.516 |
| `on_mouseover` | HTML / JavaScript | -1, 1 | 0.021 | 0.486 |
| `rightclick` | HTML / JavaScript | -1, 1 | 0.000 | 0.496 |
| `popupwindow` | HTML / JavaScript | -1, 1 | 0.000 | 0.528 |
| `iframe` | HTML / JavaScript | -1, 1 | 0.000 | 0.508 |
| `age_of_domain` | Domain / reputation | -1, 1 | 0.064 | 0.548 |
| `dnsrecord` | Domain / reputation | -1, 1 | 0.040 | 0.499 |
| `web_traffic` | Domain / reputation | -1, 0, 1 | 0.327 | 0.718 |
| `page_rank` | Domain / reputation | -1, 1 | 0.053 | 0.514 |
| `google_index` | Domain / reputation | -1, 1 | 0.136 | 0.540 |
| `links_pointing_to_page` | Domain / reputation | -1, 0, 1 | 0.124 | 0.583 |
| `statistical_report` | Domain / reputation | -1, 1 | 0.109 | 0.548 |

## Details

### `having_ip_address` — IP address used as the domain

* **Group:** URL / address bar
* **Documented rule:** Domain part of the URL is an IP address (e.g. http://125.98.3.123/...) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — IP address in URL (phishing-like): 2504 cleaned rows; phishing rate in train 57.1%
  * `1` — normal domain name: 3217 cleaned rows; phishing rate in train 47.4%
* **Single-feature validation ROC-AUC:** 0.554 [0.522, 0.586]

### `url_length` — URL length

* **Group:** URL / address bar
* **Documented rule:** Length < 54 → legitimate; 54–75 → suspicious; > 75 → phishing.
* **Values (conventional reading → observed):**
  * `-1` — long URL (> 75 chars): 4558 cleaned rows; phishing rate in train 53.2%
  * `0` — medium (54–75 chars): 96 cleaned rows; phishing rate in train 60.7%
  * `1` — short (< 54 chars): 1067 cleaned rows; phishing rate in train 44.0%
* **Single-feature validation ROC-AUC:** 0.545 [0.516, 0.574]
* **Limitations / ambiguity:** Level 0 is rare after cleaning (96 rows).

### `shortining_service` — URL shortening service (e.g. bit.ly)

* **Group:** URL / address bar
* **Documented rule:** URL uses a shortening service such as TinyURL → phishing.
* **Values (conventional reading → observed):**
  * `-1` — shortened URL (phishing-like): 814 cleaned rows; phishing rate in train 42.5%
  * `1` — not shortened: 4907 cleaned rows; phishing rate in train 53.1%
* **Single-feature validation ROC-AUC:** 0.542 [0.518, 0.564]
* **Limitations / ambiguity:** Spelling 'shortining' is from the source. Observed association with the label is weak and slightly inverted.

### `having_at_symbol` — '@' symbol in URL

* **Group:** URL / address bar
* **Documented rule:** URL contains '@' (browser ignores everything before it) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — contains '@': 1199 cleaned rows; phishing rate in train 56.3%
  * `1` — no '@': 4522 cleaned rows; phishing rate in train 50.4%
* **Single-feature validation ROC-AUC:** 0.503 [0.479, 0.529]

### `double_slash_redirecting` — '//' redirect inside the URL path

* **Group:** URL / address bar
* **Documented rule:** Last occurrence of '//' is beyond position 7 → phishing.
* **Values (conventional reading → observed):**
  * `-1` — '//' inside path: 820 cleaned rows; phishing rate in train 48.0%
  * `1` — no extra '//': 4901 cleaned rows; phishing rate in train 52.3%
* **Single-feature validation ROC-AUC:** 0.529 [0.507, 0.551]

### `prefix_suffix` — Dash '-' in the domain name

* **Group:** URL / address bar
* **Documented rule:** Domain name contains '-' (e.g. confirme-paypal.com) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — dash present (phishing-like): 5020 cleaned rows; phishing rate in train 58.8%
  * `1` — no dash: 701 cleaned rows; phishing rate in train 0.0%
* **Single-feature validation ROC-AUC:** 0.639 [0.617, 0.658]
* **Limitations / ambiguity:** All 4,898 raw phishing rows have -1, yet -1 is the majority code (87% of rows). This contradicts the document's description of '-' as rare; possible coding or collection artifact.

### `having_sub_domain` — Number of sub-domains

* **Group:** URL / address bar
* **Documented rule:** Dots in domain (after removing www. and ccTLD): 1 → legitimate; 2 → suspicious; more → phishing.
* **Values (conventional reading → observed):**
  * `-1` — multiple sub-domains: 1671 cleaned rows; phishing rate in train 62.1%
  * `0` — one sub-domain: 2014 cleaned rows; phishing rate in train 70.1%
  * `1` — no sub-domain: 2036 cleaned rows; phishing rate in train 25.4%
* **Single-feature validation ROC-AUC:** 0.692 [0.658, 0.727]
* **Limitations / ambiguity:** Non-monotone: code 0 has a higher phishing rate than -1.

### `sslfinal_state` — HTTPS certificate status

* **Group:** URL / address bar
* **Documented rule:** HTTPS with trusted issuer and certificate age ≥ 1 year → legitimate; HTTPS with untrusted issuer → suspicious; otherwise → phishing.
* **Values (conventional reading → observed):**
  * `-1` — no HTTPS / invalid: 2076 cleaned rows; phishing rate in train 88.9%
  * `0` — HTTPS, untrusted issuer: 721 cleaned rows; phishing rate in train 99.4%
  * `1` — HTTPS, trusted & ≥ 1 year: 2924 cleaned rows; phishing rate in train 14.0%
* **Single-feature validation ROC-AUC:** 0.898 [0.878, 0.917]
* **Limitations / ambiguity:** Strongest single predictor. Non-monotone: code 0 (99% phishing) is riskier than -1. The document text mentions both 1-year and 2-year certificate ages.

### `domain_registration_length` — Domain registration length

* **Group:** URL / address bar
* **Documented rule:** Domain expires in ≤ 1 year → phishing.
* **Values (conventional reading → observed):**
  * `-1` — expires within 1 year: 3737 cleaned rows; phishing rate in train 43.5%
  * `1` — registered for longer: 1984 cleaned rows; phishing rate in train 66.6%
* **Single-feature validation ROC-AUC:** 0.599 [0.566, 0.630]
* **Limitations / ambiguity:** Observed association is the reverse of the rule: -1 is associated with LEGITIMATE sites (43% vs 67% phishing). Cannot be resolved from the documentation.

### `favicon` — Favicon loaded from external domain

* **Group:** URL / address bar
* **Documented rule:** Favicon loaded from a different domain than the address bar → phishing.
* **Values (conventional reading → observed):**
  * `-1` — external favicon: 1223 cleaned rows; phishing rate in train 50.1%
  * `1` — same-domain favicon: 4498 cleaned rows; phishing rate in train 52.1%
* **Single-feature validation ROC-AUC:** 0.530 [0.502, 0.557]
* **Limitations / ambiguity:** Strongly co-occurs with popupwindow (Cramér's V 0.93) and port; near-zero label association.

### `port` — Non-standard port status

* **Group:** URL / address bar
* **Documented rule:** Server ports differ from their preferred open/closed status → phishing.
* **Values (conventional reading → observed):**
  * `-1` — non-preferred port status: 943 cleaned rows; phishing rate in train 54.3%
  * `1` — preferred port status: 4778 cleaned rows; phishing rate in train 51.1%
* **Single-feature validation ROC-AUC:** 0.488 [0.464, 0.514]
* **Limitations / ambiguity:** The document's rule text is self-contradictory ('of the preferred status → phishing').

### `https_token` — 'https' token in the domain part

* **Group:** URL / address bar
* **Documented rule:** Domain part contains the text 'https' (e.g. https-www-paypal...) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — 'https' in domain text: 1009 cleaned rows; phishing rate in train 46.8%
  * `1` — no token: 4712 cleaned rows; phishing rate in train 52.7%
* **Single-feature validation ROC-AUC:** 0.530 [0.505, 0.555]

### `request_url` — External objects (images, media)

* **Group:** Abnormal / page content
* **Documented rule:** Share of embedded objects loaded from other domains: < 22% → legitimate; otherwise phishing (document also mentions a 22–61% suspicious band).
* **Values (conventional reading → observed):**
  * `-1` — many external objects: 2535 cleaned rows; phishing rate in train 64.9%
  * `1` — mostly same-domain objects: 3186 cleaned rows; phishing rate in train 41.0%
* **Single-feature validation ROC-AUC:** 0.639 [0.604, 0.668]

### `url_of_anchor` — Anchor (<a>) links to other domains or nowhere

* **Group:** Abnormal / page content
* **Documented rule:** % of anchors pointing elsewhere or to '#'/javascript: < 31% → legitimate; 31–67% → suspicious; > 67% → phishing.
* **Values (conventional reading → observed):**
  * `-1` — > 67% (phishing-like): 2002 cleaned rows; phishing rate in train 99.1%
  * `0` — 31–67%: 2529 cleaned rows; phishing rate in train 33.8%
  * `1` — < 31%: 1190 cleaned rows; phishing rate in train 9.3%
* **Single-feature validation ROC-AUC:** 0.858 [0.836, 0.880]

### `links_in_tags` — Links in <meta>, <script>, <link> tags

* **Group:** Abnormal / page content
* **Documented rule:** % of these links to other domains: < 17% → legitimate; 17–81% → suspicious; > 81% → phishing.
* **Values (conventional reading → observed):**
  * `-1` — > 81%: 2167 cleaned rows; phishing rate in train 67.5%
  * `0` — 17–81%: 2200 cleaned rows; phishing rate in train 45.1%
  * `1` — < 17%: 1354 cleaned rows; phishing rate in train 37.1%
* **Single-feature validation ROC-AUC:** 0.655 [0.618, 0.687]

### `sfh` — Server Form Handler (form action)

* **Group:** Abnormal / page content
* **Documented rule:** Form action empty or 'about:blank' → phishing; points to a different domain → suspicious; otherwise legitimate.
* **Values (conventional reading → observed):**
  * `-1` — empty / about:blank: 4255 cleaned rows; phishing rate in train 59.6%
  * `0` — different domain: 516 cleaned rows; phishing rate in train 41.0%
  * `1` — same domain: 950 cleaned rows; phishing rate in train 22.5%
* **Single-feature validation ROC-AUC:** 0.634 [0.606, 0.659]

### `submitting_to_email` — Form submits to e-mail

* **Group:** Abnormal / page content
* **Documented rule:** Uses mail() or mailto: to submit user information → phishing.
* **Values (conventional reading → observed):**
  * `-1` — submits to e-mail: 1243 cleaned rows; phishing rate in train 51.8%
  * `1` — does not: 4478 cleaned rows; phishing rate in train 51.6%
* **Single-feature validation ROC-AUC:** 0.483 [0.457, 0.511]

### `abnormal_url` — Host name missing from URL (WHOIS)

* **Group:** Abnormal / page content
* **Documented rule:** Host name from WHOIS not included in the URL → phishing.
* **Values (conventional reading → observed):**
  * `-1` — identity not in URL: 937 cleaned rows; phishing rate in train 41.6%
  * `1` — identity in URL: 4784 cleaned rows; phishing rate in train 53.6%
* **Single-feature validation ROC-AUC:** 0.537 [0.511, 0.561]

### `redirect` — Number of redirects

* **Group:** HTML / JavaScript
* **Documented rule:** Redirects ≤ 1 → legitimate; 2–3 → suspicious; ≥ 4 → phishing.
* **Values (conventional reading → observed):**
  * `0` — code 0 (meaning undocumented): 4954 cleaned rows; phishing rate in train 50.7%
  * `1` — code 1 (meaning undocumented): 767 cleaned rows; phishing rate in train 57.5%
* **Single-feature validation ROC-AUC:** 0.516 [0.494, 0.540]
* **Limitations / ambiguity:** Coded {0, 1}, not {-1, 1}; which redirect band each code represents is undocumented.

### `on_mouseover` — onMouseOver changes the status bar

* **Group:** HTML / JavaScript
* **Documented rule:** JavaScript onMouseOver alters the status bar (fake link display) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — status bar altered: 810 cleaned rows; phishing rate in train 54.9%
  * `1` — not altered: 4911 cleaned rows; phishing rate in train 51.1%
* **Single-feature validation ROC-AUC:** 0.486 [0.463, 0.509]

### `rightclick` — Right-click disabled

* **Group:** HTML / JavaScript
* **Documented rule:** Page disables right-click (event.button==2) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — right-click disabled: 287 cleaned rows; phishing rate in train 54.8%
  * `1` — enabled: 5434 cleaned rows; phishing rate in train 51.5%
* **Single-feature validation ROC-AUC:** 0.496 [0.481, 0.511]

### `popupwindow` — Pop-up window with text fields

* **Group:** HTML / JavaScript
* **Documented rule:** Pop-up window asks for information in text fields → phishing.
* **Values (conventional reading → observed):**
  * `-1` — pop-up with text fields: 1275 cleaned rows; phishing rate in train 50.3%
  * `1` — no such pop-up: 4446 cleaned rows; phishing rate in train 52.0%
* **Single-feature validation ROC-AUC:** 0.528 [0.500, 0.554]

### `iframe` — Invisible iframe

* **Group:** HTML / JavaScript
* **Documented rule:** Page uses an iframe (typically without border) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — uses iframe: 627 cleaned rows; phishing rate in train 51.2%
  * `1` — no iframe: 5094 cleaned rows; phishing rate in train 51.7%
* **Single-feature validation ROC-AUC:** 0.508 [0.488, 0.528]

### `age_of_domain` — Age of domain

* **Group:** Domain / reputation
* **Documented rule:** Domain age ≥ 6 months (WHOIS) → legitimate; otherwise phishing.
* **Values (conventional reading → observed):**
  * `-1` — younger than 6 months: 2806 cleaned rows; phishing rate in train 55.0%
  * `1` — 6 months or older: 2915 cleaned rows; phishing rate in train 48.4%
* **Single-feature validation ROC-AUC:** 0.548 [0.514, 0.580]

### `dnsrecord` — DNS record exists

* **Group:** Domain / reputation
* **Documented rule:** No DNS/WHOIS record for the domain → phishing.
* **Values (conventional reading → observed):**
  * `-1` — no DNS record: 1513 cleaned rows; phishing rate in train 55.2%
  * `1` — DNS record found: 4208 cleaned rows; phishing rate in train 50.4%
* **Single-feature validation ROC-AUC:** 0.499 [0.470, 0.529]

### `web_traffic` — Website traffic rank (Alexa)

* **Group:** Domain / reputation
* **Documented rule:** Alexa rank < 100,000 → legitimate; > 100,000 → suspicious; no rank → phishing.
* **Values (conventional reading → observed):**
  * `-1` — no traffic rank: 1463 cleaned rows; phishing rate in train 64.3%
  * `0` — rank > 100,000: 1487 cleaned rows; phishing rate in train 70.4%
  * `1` — rank < 100,000: 2771 cleaned rows; phishing rate in train 35.0%
* **Single-feature validation ROC-AUC:** 0.718 [0.685, 0.748]
* **Limitations / ambiguity:** Non-monotone: code 0 has a higher phishing rate than -1. Alexa rankings were retired in 2022.

### `page_rank` — Google PageRank

* **Group:** Domain / reputation
* **Documented rule:** PageRank < 0.2 → phishing.
* **Values (conventional reading → observed):**
  * `-1` — PageRank < 0.2: 4056 cleaned rows; phishing rate in train 53.4%
  * `1` — PageRank ≥ 0.2: 1665 cleaned rows; phishing rate in train 47.2%
* **Single-feature validation ROC-AUC:** 0.514 [0.485, 0.547]
* **Limitations / ambiguity:** Public PageRank values are no longer published.

### `google_index` — Indexed by Google

* **Group:** Domain / reputation
* **Documented rule:** Page not in Google's index → phishing.
* **Values (conventional reading → observed):**
  * `-1` — not indexed: 985 cleaned rows; phishing rate in train 66.9%
  * `1` — indexed: 4736 cleaned rows; phishing rate in train 48.6%
* **Single-feature validation ROC-AUC:** 0.540 [0.515, 0.566]

### `links_pointing_to_page` — External links pointing to the page

* **Group:** Domain / reputation
* **Documented rule:** 0 links → phishing; 1–2 → suspicious; more → legitimate.
* **Values (conventional reading → observed):**
  * `-1` — no links: 317 cleaned rows; phishing rate in train 42.4%
  * `0` — 1–2 links: 3093 cleaned rows; phishing rate in train 57.5%
  * `1` — more than 2 links: 2311 cleaned rows; phishing rate in train 45.2%
* **Single-feature validation ROC-AUC:** 0.583 [0.549, 0.615]

### `statistical_report` — Listed in phishing statistics reports

* **Group:** Domain / reputation
* **Documented rule:** Host in PhishTank top-10 domains/IPs or StopBadware top-50 IPs (2010–2012) → phishing.
* **Values (conventional reading → observed):**
  * `-1` — listed: 884 cleaned rows; phishing rate in train 64.7%
  * `1` — not listed: 4837 cleaned rows; phishing rate in train 49.3%
* **Single-feature validation ROC-AUC:** 0.548 [0.525, 0.574]

## General limitations

* All 30 indicators are pre-discretised by the dataset authors; the thresholds (e.g. 54/75 characters, 22%/61%) were chosen on their own data and cannot be re-derived here.
* Several indicators depend on external services as of c. 2012 (WHOIS, Alexa rank, Google PageRank, PhishTank / StopBadware top lists). Some no longer exist, so the features cannot be reproduced exactly for current websites.
* Strong co-occurrence between unrelated checks (e.g. `favicon`–`popupwindow`, Cramér's V 0.93) suggests that the extraction or collection process shaped the data; this cannot be verified from the published files.
