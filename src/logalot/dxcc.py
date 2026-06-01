"""DXCC / ITU callsign-prefix → country resolution. Stdlib-only data.

A pragmatic table of the standard ITU international call-sign series, used to
*enrich* a candidate with the worked station's country and as a soft signal that
a prefix is real. It is deliberately used for enrichment only — never to reject a
callsign — so gaps here can only mean "no country shown", never a dropped call.

Resolution is longest-prefix-match: keys are 1–3 characters and the longest
matching key wins (so ``EA8`` → Canary Is. beats ``EA`` → Spain, and ``HB0`` →
Liechtenstein beats ``HB`` → Switzerland).

Coverage is the common entities (Europe in full, major DX worldwide); it is not
every DXCC exception. Grow it as needed — it is plain data.
"""
from __future__ import annotations

# Prefix -> entity. Order is irrelevant; resolution tries 3-, then 2-, then
# 1-char prefixes, so list whatever length disambiguates an entity.
_PREFIXES: dict[str, str] = {
    # --- British Isles (2-char before the G/M catch-alls) -------------------
    "GW": "Wales", "MW": "Wales", "2W": "Wales",
    "GM": "Scotland", "MM": "Scotland", "2M": "Scotland",
    "GI": "Northern Ireland", "MI": "Northern Ireland", "2I": "Northern Ireland",
    "GD": "Isle of Man", "MD": "Isle of Man",
    "GJ": "Jersey", "MJ": "Jersey",
    "GU": "Guernsey", "MU": "Guernsey",
    "G": "England", "M": "England", "2E": "England",
    "EI": "Ireland", "EJ": "Ireland",
    # --- Western Europe -----------------------------------------------------
    "F": "France", "TM": "France", "TK": "Corsica",
    "DA": "Germany", "DB": "Germany", "DC": "Germany", "DD": "Germany",
    "DE": "Germany", "DF": "Germany", "DG": "Germany", "DH": "Germany",
    "DJ": "Germany", "DK": "Germany", "DL": "Germany", "DM": "Germany",
    "DN": "Germany", "DO": "Germany", "DP": "Germany", "DQ": "Germany",
    "DR": "Germany",
    "OE": "Austria",
    "HB0": "Liechtenstein", "HB": "Switzerland", "HE": "Switzerland",
    "PA": "Netherlands", "PB": "Netherlands", "PC": "Netherlands",
    "PD": "Netherlands", "PE": "Netherlands", "PF": "Netherlands",
    "PG": "Netherlands", "PH": "Netherlands", "PI": "Netherlands",
    "ON": "Belgium", "OO": "Belgium", "OP": "Belgium", "OQ": "Belgium",
    "OR": "Belgium", "OS": "Belgium", "OT": "Belgium",
    "LX": "Luxembourg",
    "C3": "Andorra", "3A": "Monaco", "HV": "Vatican", "1A": "S.M.O.M.",
    # --- Italy / Iberia -----------------------------------------------------
    "IS0": "Sardinia", "I": "Italy",
    "EA6": "Balearic Is.", "EA8": "Canary Is.", "EA9": "Ceuta & Melilla",
    "EA": "Spain", "EB": "Spain", "EC": "Spain", "ED": "Spain", "EE": "Spain",
    "EF": "Spain", "EG": "Spain", "EH": "Spain",
    "CT3": "Madeira", "CR3": "Madeira", "CU": "Azores", "CT": "Portugal",
    # --- Nordics ------------------------------------------------------------
    "OH0": "Aland Is.", "OH": "Finland", "OF": "Finland", "OG": "Finland",
    "OI": "Finland", "OJ0": "Market Reef",
    "SM": "Sweden", "SA": "Sweden", "SB": "Sweden", "SC": "Sweden",
    "SD": "Sweden", "SE": "Sweden", "SF": "Sweden", "SG": "Sweden",
    "SH": "Sweden", "SI": "Sweden", "SJ": "Sweden", "SK": "Sweden",
    "7S": "Sweden", "8S": "Sweden",
    "JW": "Svalbard", "JX": "Jan Mayen",
    "LA": "Norway", "LB": "Norway", "LC": "Norway", "LN": "Norway",
    "OX": "Greenland", "OY": "Faroe Is.",
    "OZ": "Denmark", "5P": "Denmark", "5Q": "Denmark",
    "TF": "Iceland",
    # --- Baltics & Central/Eastern Europe -----------------------------------
    "ES": "Estonia", "YL": "Latvia", "LY": "Lithuania",
    "SP": "Poland", "SN": "Poland", "SO": "Poland", "SQ": "Poland",
    "SR": "Poland", "HF": "Poland", "3Z": "Poland",
    "OK": "Czechia", "OL": "Czechia", "OM": "Slovakia",
    "HA": "Hungary", "HG": "Hungary",
    "YO": "Romania", "YP": "Romania", "YQ": "Romania", "YR": "Romania",
    "LZ": "Bulgaria",
    "S5": "Slovenia", "9A": "Croatia", "E7": "Bosnia-Herzegovina",
    "YU": "Serbia", "YT": "Serbia", "YZ": "Serbia",
    "Z3": "North Macedonia", "Z6": "Kosovo", "ZA": "Albania",
    "4O": "Montenegro",
    # --- Greece / Mediterranean ---------------------------------------------
    "SV5": "Dodecanese", "SV9": "Crete", "SY": "Mount Athos",
    "SV": "Greece", "SW": "Greece", "SX": "Greece", "SZ": "Greece", "J4": "Greece",
    "9H": "Malta", "5B": "Cyprus", "C4": "Cyprus", "H2": "Cyprus", "P3": "Cyprus",
    "TA": "Turkey", "TB": "Turkey", "TC": "Turkey",
    # --- Eastern Europe / Caucasus ------------------------------------------
    "UR": "Ukraine", "US": "Ukraine", "UT": "Ukraine", "UU": "Ukraine",
    "UV": "Ukraine", "UW": "Ukraine", "UX": "Ukraine", "UY": "Ukraine",
    "UZ": "Ukraine", "EM": "Ukraine", "EN": "Ukraine", "EO": "Ukraine",
    "EU": "Belarus", "EV": "Belarus", "EW": "Belarus",
    "ER": "Moldova", "4L": "Georgia", "EK": "Armenia",
    "4J": "Azerbaijan", "4K": "Azerbaijan",
    "R": "Russia", "UA": "Russia", "UB": "Russia", "UC": "Russia",
    "UD": "Russia", "UE": "Russia", "UF": "Russia", "UG": "Russia",
    "UH": "Russia", "UI": "Russia",
    "UN": "Kazakhstan", "UK": "Uzbekistan", "EX": "Kyrgyzstan",
    "EY": "Tajikistan", "EZ": "Turkmenistan",
    # --- North America ------------------------------------------------------
    "KH6": "Hawaii", "KL": "Alaska", "KP2": "US Virgin Is.", "KP4": "Puerto Rico",
    "K": "United States", "N": "United States", "W": "United States",
    "AA": "United States", "AB": "United States", "AC": "United States",
    "AD": "United States", "AE": "United States", "AF": "United States",
    "AG": "United States", "AI": "United States", "AJ": "United States",
    "AK": "United States",
    "VE": "Canada", "VA": "Canada", "VO": "Canada", "VY": "Canada",
    "CF": "Canada", "CG": "Canada", "CH": "Canada", "CI": "Canada",
    "CJ": "Canada", "CK": "Canada",
    "XE": "Mexico", "XF": "Mexico", "XA": "Mexico", "XB": "Mexico",
    "XC": "Mexico", "XD": "Mexico", "XI": "Mexico",
    # --- Central / South America --------------------------------------------
    "PY": "Brazil", "PP": "Brazil", "PQ": "Brazil", "PR": "Brazil",
    "PS": "Brazil", "PT": "Brazil", "PU": "Brazil", "PV": "Brazil",
    "PW": "Brazil", "ZV": "Brazil", "ZW": "Brazil", "ZX": "Brazil",
    "ZY": "Brazil", "ZZ": "Brazil",
    "LU": "Argentina", "CE": "Chile", "HK": "Colombia", "OA": "Peru",
    "YV": "Venezuela", "CX": "Uruguay", "HC": "Ecuador", "CP": "Bolivia",
    "ZP": "Paraguay", "HI": "Dominican Rep.", "CO": "Cuba", "CM": "Cuba",
    # --- East Asia ----------------------------------------------------------
    "JA": "Japan", "JE": "Japan", "JF": "Japan", "JG": "Japan", "JH": "Japan",
    "JI": "Japan", "JJ": "Japan", "JK": "Japan", "JL": "Japan", "JM": "Japan",
    "JN": "Japan", "JO": "Japan", "JP": "Japan", "JQ": "Japan", "JR": "Japan",
    "JS": "Japan", "7J": "Japan", "7K": "Japan", "7L": "Japan", "7M": "Japan",
    "7N": "Japan", "8J": "Japan", "8N": "Japan",
    "BV": "Taiwan", "BY": "China", "BA": "China", "BD": "China", "BG": "China",
    "BH": "China", "BI": "China", "BT": "China",
    "VR": "Hong Kong", "XX9": "Macao",
    "HL": "South Korea", "DS": "South Korea", "6K": "South Korea",
    "6L": "South Korea", "6M": "South Korea", "6N": "South Korea",
    "P5": "North Korea",
    # --- South / SE Asia ----------------------------------------------------
    "VU": "India", "AP": "Pakistan", "S2": "Bangladesh", "4S": "Sri Lanka",
    "9N": "Nepal", "HS": "Thailand", "E2": "Thailand", "XV": "Vietnam",
    "3W": "Vietnam", "9M": "Malaysia", "9W": "Malaysia", "9V": "Singapore",
    "YB": "Indonesia", "YC": "Indonesia", "YD": "Indonesia", "YE": "Indonesia",
    "YF": "Indonesia", "YG": "Indonesia", "YH": "Indonesia",
    "DU": "Philippines", "DV": "Philippines", "DW": "Philippines",
    "DX": "Philippines", "DY": "Philippines", "DZ": "Philippines",
    # --- Middle East --------------------------------------------------------
    "4X": "Israel", "4Z": "Israel", "JY": "Jordan", "OD": "Lebanon",
    "YK": "Syria", "YI": "Iraq", "EP": "Iran", "EQ": "Iran",
    "A4": "Oman", "A6": "United Arab Emirates", "A7": "Qatar", "A9": "Bahrain",
    "9K": "Kuwait", "HZ": "Saudi Arabia", "7Z": "Saudi Arabia", "8Z": "Saudi Arabia",
    "A2": "Botswana",
    # --- Oceania ------------------------------------------------------------
    "VK": "Australia", "ZL": "New Zealand", "FK": "New Caledonia",
    "FO": "French Polynesia", "KH2": "Guam", "3D2": "Fiji",
    # --- Africa -------------------------------------------------------------
    "ZS": "South Africa", "ZR": "South Africa", "ZT": "South Africa",
    "ZU": "South Africa", "SU": "Egypt", "CN": "Morocco", "7X": "Algeria",
    "3V": "Tunisia", "5A": "Libya", "5N": "Nigeria", "5Z": "Kenya",
    "5H": "Tanzania", "9G": "Ghana", "9J": "Zambia", "Z2": "Zimbabwe",
    "D2": "Angola", "C9": "Mozambique", "V5": "Namibia", "EL": "Liberia",
    "5R": "Madagascar", "3B8": "Mauritius", "FR": "Reunion",
}


def _core_prefix_source(call: str) -> str:
    """Uppercased call without any portable affix, for prefix matching."""
    c = call.upper().strip()
    return c.split("/", 1)[0] if "/" in c else c


def country_for_call(call: str) -> str | None:
    """Resolve a callsign to its DXCC country, or None if the prefix is unknown."""
    if not call:
        return None
    c = _core_prefix_source(call)
    for n in (3, 2, 1):
        if len(c) >= n:
            entity = _PREFIXES.get(c[:n])
            if entity is not None:
                return entity
    return None


def has_known_prefix(call: str) -> bool:
    return country_for_call(call) is not None
