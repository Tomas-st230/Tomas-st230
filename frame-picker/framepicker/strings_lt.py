"""Visos lietuviškos naudotojui rodomos eilutės.

This is the ONLY module in the project that may contain Lithuanian text.
``tests/test_no_hardcoded_lt_strings.py`` enforces that rule.

Every entry is either a plain constant or a function that formats already
measured numbers into a sentence. Nothing here computes anything.
"""

from __future__ import annotations

APP_TITLE = "Geriausių kadrų rinkiklis"

# --------------------------------------------------------------------------
# Eiga ir santrauka
# --------------------------------------------------------------------------


def processing_file(index: int, total: int, name: str) -> str:
    return f"[{index}/{total}] Apdorojama: {name}"


def probe_failed(name: str, detail: str) -> str:
    return f"Failo nepavyko nuskaityti, praleidžiama: {name} ({detail})"


def decode_path_hw() -> str:
    return "Dekodavimas: aparatinis (CUDA)"


def decode_path_cpu(reason: str = "") -> str:
    if reason:
        return f"Dekodavimas: programinis (CPU), nes aparatinis nesuveikė: {reason}"
    return "Dekodavimas: programinis (CPU)"


def sampled_frames(got: int, expected: int, fps: float) -> str:
    return f"Išrinkta kandidatų: {got} (tikėtasi ~{expected}, kai imama {fps:g} kadro per sekundę)"


def sampling_shortfall(got: int, expected: int) -> str:
    return (
        f"DĖMESIO: gauta mažiau kandidatų, nei tikėtasi pagal trukmę "
        f"({got} vietoj ~{expected}). Įraše gali būti nutrūkusių ar sugadintų kadrų."
    )


def rejected_frames(count: int, dark: int, bright: int, blurry: int) -> str:
    return (
        f"Atmesta per pigų patikrinimą: {count} "
        f"(vien juoda: {dark}, vien perdegę: {bright}, neryškūs: {blurry})"
    )


def clip_done(name: str, chosen: int, requested: int, seconds: float) -> str:
    return f"Baigta: {name} — parinkta {chosen} iš prašytų {requested} kadrų, užtruko {seconds:.1f} s"


def clip_done_threshold(name: str, chosen: int, seconds: float) -> str:
    return f"Baigta: {name} — ribą peržengė ir buvo parinkta {chosen} kadrų, užtruko {seconds:.1f} s"


def throughput(seconds_per_footage_minute: float) -> str:
    return f"Išmatuotas našumas: {seconds_per_footage_minute:.1f} s vienai medžiagos minutei šiame kompiuteryje"


def batch_summary(
    files_given: int, files_done: int, files_skipped: int, frames_requested: int, frames_delivered: int
) -> str:
    return (
        f"Santrauka: pateikta failų {files_given}, apdorota {files_done}, praleista {files_skipped}; "
        f"prašyta kadrų {frames_requested}, pateikta {frames_delivered}"
    )


def batch_summary_threshold(files_given: int, files_done: int, files_skipped: int, frames_delivered: int) -> str:
    return (
        f"Santrauka: pateikta failų {files_given}, apdorota {files_done}, praleista {files_skipped}; "
        f"pateikta kadrų {frames_delivered} (kiekis priklauso nuo kokybės ribos, nebuvo prašoma fiksuoto skaičiaus)"
    )


def skipped_files_header() -> str:
    return "Praleisti failai:"


def output_written(path: str) -> str:
    return f"Rezultatai įrašyti: {path}"


def cancelled_cleanup(path: str) -> str:
    return f"Atšaukta. Nebaigti failai ištrinti: {path}"


def no_input_files() -> str:
    return "Nenurodytas nė vienas vaizdo failas."


def input_not_found(path: str) -> str:
    return f"Pagal nurodymą nerasta nė vieno vaizdo failo: {path}"


def ordered_by_date() -> str:
    return "Failai apdorojami nuo seniausio iki naujausio (pagal failo datą diske)"


def ordered_by_name() -> str:
    return "Failai apdorojami pagal pavadinimą"


def inputs_expanded(given: int, found: int) -> str:
    return f"Nurodyta įvesčių: {given}; rasta vaizdo failų: {found}"


def ffmpeg_missing(name: str) -> str:
    return f"Sistemoje nerastas {name}. Įdiekite ffmpeg ir įtraukite jį į PATH."


def keep_awake_on() -> str:
    return (
        "Kol vyksta apdorojimas, kompiuteriui neleidžiama užmigti savaime. "
        "Išjungti, perkrauti, atsijungti ar užmigdyti ranka galima bet kada — "
        "ekranas taip pat gali gesti ir kompiuteris užsirakinti kaip įprastai."
    )


def keep_awake_unavailable(detail: str) -> str:
    return f"Miego blokavimas šioje sistemoje neveikia ({detail}). Apdorojimas tęsiamas."


def unknown_duration() -> str:
    return "trukmė nežinoma"


# --------------------------------------------------------------------------
# Plokščio (log) profilio aptikimas
# --------------------------------------------------------------------------


def log_detected(source: str) -> str:
    return f"Aptiktas plokščias (log) profilis. Pagrindas: {source}"


def log_not_detected(source: str) -> str:
    return f"Plokščio (log) profilio neaptikta. Pagrindas: {source}"


LOG_SOURCE_FLAG = "naudotojo nurodyta vėliavėlė --convert-log"
LOG_SOURCE_METADATA = "spalvų metaduomenys faile"
LOG_SOURCE_FILENAME = "užuomina failo pavadinime"
LOG_SOURCE_STATISTICS = "išmatuotas kadrų plokštumas"
LOG_SOURCE_DEFAULT = "numatytoji reikšmė (išjungta)"


def log_statistics(luma_span: float, saturation: float) -> str:
    return f"Išmatuota iš kadrų: šviesumo diapazonas {luma_span:.3f}, sodrumas {saturation:.3f}"


def log_suspected_not_applied(luma_span: float, saturation: float) -> str:
    return (
        f"ĮTARIAMA plokščia medžiaga (šviesumo diapazonas {luma_span:.3f}, sodrumas {saturation:.3f}), "
        f"BET jokia transformacija netaikyta. Išmatuotas plokštumas rodo sceną, ne profilį: "
        f"tamsus saulėlydis matuojasi taip pat kaip log. Jei tai tikrai log, nurodykite "
        f"--convert-log on arba lange pasirinkite „visi failai yra log“."
    )


def log_thresholds(max_span: float, max_saturation: float) -> str:
    return (
        f"Įtarimo ribos: diapazonas < {max_span:.2f} IR sodrumas < {max_saturation:.2f}. "
        f"Šios ribos NIEKO nekeičia — tik įrašo įtarimą į ataskaitą."
    )


def log_is_a_guess() -> str:
    return (
        "Tai spėjimas, o ne patikrintas faktas: profilis nustatytas pagal "
        "metaduomenis arba failo pavadinimą."
    )


def lut_applied(path: str) -> str:
    return f"Analizei pritaikytas LUT: {path}"


def normalisation_applied(strength: float, gain: float) -> str:
    return (
        f"Pritaikytas procentilinis kontrasto ir sodrumo normalizavimas "
        f"(stiprumas {strength:.2f}, sodrumo daugiklis {gain:.2f}). "
        f"Tai APYTIKSLIS perskaičiavimas, o ne spalvų valdymo grandinė — "
        f"skaičiai nėra kolorimetriškai teisingi. Tikram log naudokite LUT."
    )


def no_conversion_applied() -> str:
    return "Analizei jokia spalvų transformacija netaikyta."


def lut_skipped_not_log() -> str:
    return (
        "LUT nepritaikytas: šis įrašas neatpažintas kaip plokščias (log), o LUT ant "
        "įprastos Rec.709 medžiagos ją sugadintų. Priverstinai — vėliavėlė --lut-all."
    )


def lut_forced_on_all() -> str:
    return "LUT pritaikytas priverstinai visiems įrašams (--lut-all), neatsižvelgiant į profilį."


def lut_unreadable(path: str, detail: str) -> str:
    return f"LUT failo nepavyko perskaityti ({path}: {detail}). Naudojamas normalizavimas."


# --------------------------------------------------------------------------
# Degradavimo pranešimai (kiekvienas rodomas vieną kartą per paleidimą)
# --------------------------------------------------------------------------


def faces_unavailable(detail: str) -> str:
    return (
        f"Veidų aptikimas neveikia ({detail}). Veidų požymiai lieka nežinomi (None) "
        f"ir į vertinimą neįtraukiami — analizė remiasi tik statistika."
    )


def saliency_unavailable(detail: str) -> str:
    return f"Objekto išskyrimas neveikia ({detail}). Kompozicijos dedamoji praleidžiama."


STATISTICS_ONLY_MODE = "Režimas: tik statistika (be veidų aptikimo)"
FACES_DISABLED_BY_FLAG = "išjungta vėliavėle --no-faces"
FACE_MODEL_MISSING = "trūksta YuNet modelio failo"
SALIENCY_MISSING = "trūksta opencv-contrib"


# --------------------------------------------------------------------------
# Vertinimo paaiškinimai
# --------------------------------------------------------------------------


def reason_face_area(rel: float) -> str:
    return f"veidas užima {rel * 100:.0f} % kadro"


def reason_faces_count(count: int) -> str:
    return f"kadre aptikta veidų: {count}"


def reason_no_faces() -> str:
    return "veidų nerasta"


def reason_faces_unknown() -> str:
    return "veidų duomenų nėra (aptikimas neveikia)"


def reason_sharpness_rank(percentile: float, measured: float | None = None) -> str:
    text = f"ryškumas — {percentile:.0f}-as procentilis šiame įraše"
    if measured is not None:
        # Percentilis rodo vietą įraše, o šis skaičius — patį išmatuotą dydį,
        # kad kadrus būtų galima palyginti ir tarp skirtingų įrašų.
        text += f" (išmatuota {measured:.1f})"
    return text


def reason_colorfulness_rank(percentile: float) -> str:
    return f"spalvingumas — {percentile:.0f}-as procentilis šiame įraše"


def reason_dynamic_range_rank(percentile: float) -> str:
    return f"dinaminis diapazonas — {percentile:.0f}-as procentilis šiame įraše"


def reason_clipped_high(fraction: float) -> str:
    return f"{fraction * 100:.0f} % kadro perdegę"


def reason_clipped_low(fraction: float) -> str:
    return f"{fraction * 100:.0f} % kadro užgesę į juodą"


def reason_thirds(distance: float) -> str:
    return f"objektas per {distance:.2f} nuo trečdalių taško (0 = tiksliai ant jo)"


def reason_subject_size(rel: float) -> str:
    return f"pagrindinis objektas užima {rel * 100:.0f} % kadro"


def reason_composition_unknown() -> str:
    return "kompozicija nevertinta (nėra objekto išskyrimo)"


def reason_component(name: str, value: float) -> str:
    return f"{name}: {value:.2f}"


def reason_technical_gate(percentile: float) -> str:
    """Kodėl „technika“ dažnai lygi 1.00, nors ryškumo procentilis skiriasi."""
    return (
        f"technika yra riba, ne skalė: nuo {percentile:.0f}-o ryškumo procentilio "
        "balas nebekyla — už papildomą ryškumą kadras negauna nieko"
    )


def reason_moment_capped(cap: float) -> str:
    return (
        f"momentas be žmonių kadre ribojamas iki {cap:.2f} — judantis kadras be "
        "objekto vertinamas mažiau nei judantis kadras su žmogumi"
    )


def reason_horizon_tilt(degrees: float) -> str:
    return f"horizontas pakrypęs apie {degrees:.1f}°"


def reason_subject_separation(value: float) -> str:
    return f"objektas atsiskiria nuo fono: {value:.2f} (1 = labai aiškiai)"


def reason_motion(percentile: float) -> str:
    return f"judesys — {percentile:.0f}-as procentilis šiame įraše"


def reason_motion_unknown() -> str:
    return "judesio duomenų nėra (pirmas įrašo kadras)"


#: Only for turning the 0..1 tilt back into degrees inside a message.
TILT_FULL_DEGREES_FOR_TEXT = 8.0

COMPONENT_CONTENT = "turinys"
COMPONENT_TECHNICAL = "technika"
COMPONENT_COMPOSITION = "kompozicija"
COMPONENT_MOMENT = "momentas"


# --------------------------------------------------------------------------
# Pasitikėjimas reitingu (TRAP-11)
# --------------------------------------------------------------------------


def confidence_low(spread: float, n: int) -> str:
    return (
        f"MAŽAS PATIKIMUMAS: {n} geriausių kadrų įverčiai telpa į {spread:.3f} pločio "
        f"ruožą (0–1 skalėje). Reitingas beveik neneša informacijos — kadrai "
        f"praktiškai lygiaverčiai, rinkitės akimis."
    )


def confidence_ok(spread: float, n: int) -> str:
    return f"Reitingas informatyvus: {n} geriausių kadrų įverčiai išsiskiria per {spread:.3f} (0–1 skalėje)."


# --------------------------------------------------------------------------
# Atranka ir trūkumai
# --------------------------------------------------------------------------


def selection_mode_threshold(threshold: float) -> str:
    return (
        f"Atranka pagal kokybės ribą: imami kadrai, kurių įvertis ≥ {threshold:.2f}. "
        f"Riba yra pradinė, NEIŠMATUOTA reikšmė — ją reikia kalibruoti su savo medžiaga."
    )


def selection_mode_count(requested: int) -> str:
    return f"Atranka pagal kiekį: prašoma {requested} kadrų iš įrašo"


def threshold_passed(count: int, threshold: float) -> str:
    return f"Ribą {threshold:.2f} peržengė kandidatų: {count}"


def threshold_none_passed(best: float, threshold: float) -> str:
    return (
        f"Nė vienas kadras nepasiekė ribos {threshold:.2f} (geriausias įvertis {best:.3f}) — "
        f"iš šio įrašo neišsaugota nieko. Sumažinkite ribą, jeigu norite matyti geriausius turimus kadrus."
    )


def threshold_capped(kept: int) -> str:
    return f"Apribota iki {kept} kadrų (--max-per-clip); ribą peržengė daugiau kandidatų"


LOOK_NAMES = {
    "none": "be profilio",
    "auto": "automatiškai (pagal sceną)",
    "nature": "gamta",
    "city": "miestas",
}


def look_applied(name: str, strength: float) -> str:
    return (
        f"Eksportuojamiems kadrams pritaikytas spalvų profilis „{LOOK_NAMES.get(name, name)}“ "
        f"(stiprumas {strength:.2f}). Kiekvienas kadras pirma išmatuojamas ir tik tada "
        f"pastumiamas link profilio tikslų — nieko neužmetama ant viršaus vienodai. "
        f"Analizės įverčiams profilis netaikomas."
    )


def look_none() -> str:
    return "Spalvų profilis netaikytas (--look none)."


def export_resolution_native() -> str:
    return "Eksportuojama šaltinio raiška (be mažinimo)"


def export_resolution_scaled(height: int) -> str:
    return f"Eksportuojant mažinama iki {height}p (šaltinio raiška išsaugoma tik results.json)"


def shortfall_header(delivered: int, requested: int) -> str:
    return f"Pateikta {delivered} kadrų vietoj prašytų {requested}. Priežastis:"


def shortfall_clip_too_short(duration: float, gap: float, possible: int) -> str:
    return (
        f"įrašas trunka {duration:.1f} s, o mažiausias tarpas tarp kadrų yra "
        f"{gap:.1f} s — telpa daugiausia {possible} kadrai"
    )


def shortfall_near_duplicates(rejected: int) -> str:
    return f"likę kandidatai vizualiai beveik vienodi (atmesta kaip dublikatai: {rejected})"


def shortfall_not_enough_candidates(candidates: int) -> str:
    return f"po pigaus patikrinimo liko tik {candidates} kandidatai"


def shortfall_time_gap(rejected: int) -> str:
    return f"atmesta dėl per mažo laiko tarpo: {rejected}"


def export_failed(timestamp: float, detail: str) -> str:
    return f"Kadro ties {timestamp:.3f} s išsaugoti nepavyko: {detail}"


# --------------------------------------------------------------------------
# Ataskaita (report.html)
# --------------------------------------------------------------------------

def integrity_header() -> str:
    return "Galutinis patikrinimas (ar viskas vietoje ir ar visos sąsajos veikia):"


def integrity_ok(frames: int, files: int, previews: int) -> str:
    return (
        f"Viskas tvarkoje: ataskaitoje {frames} kadrų, diske rasti visi {files} failai, "
        f"įterptos visos {previews} peržiūros."
    )


def integrity_missing_files(count: int, names: str) -> str:
    return f"KLAIDA: pažymėta kaip išsaugota, bet failų diske nėra ({count}): {names}"


def integrity_empty_files(count: int, names: str) -> str:
    return f"KLAIDA: failai yra, bet tušti arba nenuskaitomi ({count}): {names}"


def integrity_missing_previews(count: int, names: str) -> str:
    return (
        f"Ataskaitoje neįsikėlė peržiūros ({count}): {names}. Pats JPEG failas aplanke yra — "
        f"nepavyko jo perskaityti peržiūrai."
    )


def integrity_failed_exports(count: int, names: str) -> str:
    return f"Nepavyko išsaugoti kadrų ({count}): {names}"


def integrity_orphans(count: int, names: str) -> str:
    return f"Aplanke yra failų, kurių ataskaita nemini ({count}): {names}"


def integrity_report_files(json_ok: bool, html_ok: bool) -> str:
    return f"results.json: {'yra' if json_ok else 'NĖRA'}; report.html: {'yra' if html_ok else 'NĖRA'}"


REPORT_INTEGRITY = "Galutinis patikrinimas"
REPORT_TITLE = "Geriausių kadrų ataskaita"
REPORT_GENERATED = "Sugeneruota"
REPORT_CLIP = "Įrašas"
REPORT_DURATION = "Trukmė"
REPORT_RESOLUTION = "Raiška"
REPORT_CODEC = "Kodekas"
REPORT_FPS = "Kadrų dažnis"
REPORT_DECODE = "Dekodavimas"
REPORT_CANDIDATES = "Įvertinta kandidatų"
REPORT_REJECTED = "Atmesta"
REPORT_ELAPSED = "Truko"
REPORT_SCORE = "Įvertis"
REPORT_TIMESTAMP = "Laikas"
REPORT_RANK = "Vieta"
REPORT_REASONS = "Kodėl"
REPORT_SKIPPED_FILES = "Praleisti failai"
REPORT_SUMMARY = "Santrauka"
REPORT_WEIGHTS = "Svoriai"
REPORT_WEIGHTS_NOTE = (
    "Svoriai yra pradinė, NEIŠMATUOTA reikšmė. Juos reikia kalibruoti su tikra "
    "medžiaga, pirma nei kuo nors pasitikėti."
)
REPORT_COLOR_NOTE = "Spalvų apdorojimas"
REPORT_MODE = "Režimas"
REPORT_SELECTION = "Atranka"
REPORT_THRESHOLD = "Kokybės riba"
REPORT_EXPORT = "Eksportas"
REPORT_GLOBAL = "Geriausi visos partijos kadrai"
REPORT_GLOBAL_NOTE = (
    "Šis palyginimas tarp skirtingų įrašų remiasi neperskaičiuotomis reikšmėmis ir "
    "yra MAŽIAU patikimas nei reitingas viename įraše."
)
REPORT_NO_FRAMES = "Nė vieno kadro parinkti nepavyko."


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

GUI_DROP_HERE = "Nutempkite vaizdo failus arba aplanką čia\n(dukart spustelėkite, kad pasirinktumėte)"
GUI_START = "Pradėti"
GUI_CANCEL = "Atšaukti"
GUI_OPEN_FOLDER = "Atverti rezultatų aplanką"
GUI_OPEN_REPORT = "Atverti ataskaitą"
GUI_CLEAR = "Išvalyti sąrašą"
GUI_IDLE = "Laukiama failų"
GUI_DONE = "Baigta"
GUI_CANCELLED = "Atšaukta"

GUI_COL_FILE = "Failas"
GUI_COL_PROFILE = "Profilis"
GUI_COL_COLOR = "Spalvos"
GUI_COL_DECODE = "Dekodavimas"
GUI_COL_FRAMES = "Kadrai"
GUI_COL_STATUS = "Būsena"

GUI_LUT = "Drono LUT (.cube)"
GUI_BROWSE = "Parinkti…"
GUI_LUT_ALL = "Taikyti visiems failams"
GUI_LUT_HINT = (
    "LUT pritaikomas tik tiems failams, kurie atpažinti kaip plokšti (D-Log M, HLG ir pan.). "
    "Įprastų Rec.709 failų partijoje jis nekeičiamas — LUT juos sugadintų."
)
GUI_PROFILE = "Plokščio profilio aptikimas"
GUI_PROFILE_AUTO = "automatiškai"
GUI_PROFILE_ON = "visi failai yra log"
GUI_PROFILE_OFF = "nė vienas nėra log"
GUI_MIN_SCORE = "Kokybės riba"
GUI_MAX_PER_CLIP = "Daugiausia kadrų iš failo"
GUI_OUT_DIR = "Rezultatų aplankas"
GUI_SETTINGS = "Nustatymai"
GUI_LOOK = "Spalvų profilis"
GUI_LOOK_STRENGTH = "Profilio stiprumas"

GUI_LOG_YES = "plokščias (log)"
GUI_LOG_NO = "įprastas"
GUI_UNKNOWN = "—"
GUI_WAITING = "laukia"
GUI_FAILED = "praleista"
GUI_COLOR_LUT = "LUT"
GUI_COLOR_NORMALISE = "normalizuota"
GUI_COLOR_NONE = "be pakeitimų"
GUI_DECODE_HW = "GPU (CUDA)"
GUI_DECODE_CPU = "CPU"


# --------------------------------------------------------------------------
# Proxy (.LRF), caption sidecar (.SRT), automatic look, per-run folder
# --------------------------------------------------------------------------


def proxy_used(name: str, width: int, height: int) -> str:
    return (
        f"Analizei naudojamas gretimas peržiūros failas {name} ({width}x{height}). "
        "Eksportuojami kadrai visada imami iš originalo."
    )


def proxy_rejected(name: str, detail: str) -> str:
    return f"Peržiūros failas {name} nenaudojamas: {detail}."


def color_mode_found(value: str, source: str) -> str:
    return f"Kameros užrašuose ({source}) nurodytas spalvų režimas: color_md={value}."


def color_mode_missing() -> str:
    return (
        "Gretimo .SRT failo su kameros užrašais nėra, todėl profilis nustatomas iš "
        "kitų požymių (bitų gylio, metaduomenų, pavadinimo). Įjunkite drone „Video Captions“, "
        "ir profilis bus žinomas tiksliai."
    )


def lut_profile_mismatch(profile: str) -> str:
    return (
        f"Šis failas atpažintas kaip {profile.upper()}, o LUT paprastai skirtas D-Log M. "
        "Spalvos gali būti netikslios — patikrinkite bent vieną kadrą."
    )


def gpu_scale_on(scaler: str) -> str:
    return (
        f"Kadrai mažinami GPU ({scaler}), todėl į atmintį kopijuojami jau maži. "
        "Tai didžiausias dekodavimo laiko taupymas 4K medžiagoje."
    )


def gpu_scale_off(detail: str) -> str:
    text = "GPU mažinimas neprieinamas, kadrai mažinami procesoriuje"
    return f"{text}: {detail}." if detail else f"{text}."


def keyframes_only(frames: int, fps: float) -> str:
    return (
        f"Dekoduoti tik atskaitos kadrai (keyframes): {frames} kadrų, "
        f"tankis apie {fps:.2f}/s. Greita, bet tinklelį nustato kamera, ne nustatymai."
    )


def look_auto_decided(name: str, nature: float, city: float, frames: int) -> str:
    return (
        f"Automatiškai parinktas profilis „{LOOK_NAMES.get(name, name)}“ "
        f"(gamta {nature:.2f} / miestas {city:.2f}, išmatuota {frames} kadr.)."
    )


def look_auto_undecided(nature: float | None, city: float | None, margin: float) -> str:
    if nature is None or city is None:
        return "Automatinis profilis nenustatytas: nebuvo ką išmatuoti. Profilis netaikomas."
    return (
        f"Automatinis profilis nenustatytas: gamta {nature:.2f} ir miestas {city:.2f} "
        f"skiriasi mažiau nei {margin:.2f}. Profilis netaikomas — tai sąmoningas atsakymas, "
        "o ne klaida."
    )


def run_folder_created(path: str) -> str:
    return f"Šio paleidimo rezultatai rašomi į atskirą aplanką: {path}"


def integrity_links(checked: int, broken: int) -> str:
    if not checked:
        return "Ataskaitos sąsajų nebuvo ko tikrinti: ataskaitoje nėra nė vieno kadro."
    if broken:
        return f"Ataskaitos sąsajos: patikrinta {checked}, neveikia {broken}."
    return f"Ataskaitos sąsajos: patikrinta {checked}, visos veikia."


GUI_LOOK_AUTO_HINT = (
    "„Automatiškai“ išmatuoja kiekvieno failo scenas (žalumas, dangus, šiluma, "
    "pilkos plokštumos, vertikalios linijos) ir parenka „gamta“ arba „miestas“. "
    "Kai požymiai per artimi, profilis netaikomas."
)
GUI_COL_LOOK = "Profilis"
GUI_STOPPING = "Stabdoma…"
GUI_RESET = "Pradėti iš naujo"
GUI_BUSY = "Vykdoma — naujų failų kelti negalima"
GUI_RUN_DIR = "Šio paleidimo aplankas"


def reason_symmetry(value: float) -> str:
    return f"simetrija — {value:.2f} (0 = nesimetriška, 1 = veidrodinė)"


def reason_pattern(value: float) -> str:
    return f"pasikartojantis raštas — {value:.2f} (laukai, stogai, bangos)"


def reason_negative_space(value: float) -> str:
    return f"tuščia erdvė aplink objektą — {value:.2f}"


def reason_no_subject_placement() -> str:
    return "objekto vietos nustatyti nepavyko, kompozicija vertinta tik pagal grafinius požymius"


# --------------------------------------------------------------------------
# Kalibravimas pagal paties pasirinktus kadrus (framepicker.learn)
# --------------------------------------------------------------------------

LEARN_TITLE = "Kalibravimas: ką atrinkote patys, palyginti su tuo, ką atrinko įrankis"
LEARN_COMPONENT_HEADER = (
    "Dedamosios (vidurkis atrinktų / vidurkis atmestų / efekto dydis d / kiek išmatuota):"
)
LEARN_WEIGHTS_HEADER = "Pasiūlyti svoriai (dabartinis → pasiūlytas):"


def learn_counts(frames: int, matched: int, given: int, dropped: int) -> str:
    return (
        f"Ataskaitoje kadrų: {frames}. Jūsų atrinkta: {given}, iš jų atpažinta {matched}. "
        f"Neatrinktų kadrų: {dropped}."
    )


def learn_unmatched(names: str) -> str:
    return f"Šių atrinktų failų ataskaitoje nėra (pervardinti ar iš kito paleidimo): {names}"


def learn_component_line(
    name: str, kept: float | None, dropped: float | None, effect: float | None,
    kept_n: int, dropped_n: int,
) -> str:
    def number(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    effect_text = "—" if effect is None else f"{effect:+.2f}"
    return f"  {name:12s} {number(kept)} / {number(dropped)} / d={effect_text} / {kept_n}+{dropped_n}"


def learn_weight_line(name: str, current: float, proposed: float) -> str:
    return f"  {name:12s} {current:.3f} → {proposed:.3f}"


def learn_not_enough(kept: int, dropped: int, min_kept: int, min_dropped: int) -> str:
    return (
        f"Svoriai neskaičiuoti: atrinkta {kept} (reikia bent {min_kept}), "
        f"atmesta {dropped} (reikia bent {min_dropped}). Iš kelių pavyzdžių gautas "
        "svoris yra atsitiktinumas, o ne kalibravimas."
    )


def learn_summary(kept: int, dropped: int) -> str:
    return f"Skaičiuota iš {kept} atrinktų ir {dropped} atmestų kadrų."


def learn_hit_rate(before: int, after: int, top_n: int) -> str:
    return (
        f"Patikra: tarp pirmų {top_n} kadrų jūsų pasirinkimų buvo {before}, "
        f"su pasiūlytais svoriais būtų {after}."
    )


def learn_no_improvement() -> str:
    return (
        "Pasiūlyti svoriai nepagerino rezultato — tai reiškia, kad jūsų pasirinkimų "
        "šios dedamosios neaiškina. Svorių keisti neverta."
    )


def learn_not_applied() -> str:
    return (
        "Svoriai NĖRA pritaikyti automatiškai. Jei norite juos naudoti, pakeiskite "
        "WEIGHTS reikšmes framepicker/scoring.py — taip sprendimas lieka žmogaus rankose."
    )


def learn_cannot_read(path: str, detail: str) -> str:
    return f"Nepavyko perskaityti {path}: {detail}"
