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


def ffmpeg_missing(name: str) -> str:
    return f"Sistemoje nerastas {name}. Įdiekite ffmpeg ir įtraukite jį į PATH."


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
LOG_SOURCE_DEFAULT = "numatytoji reikšmė (išjungta)"


def log_is_a_guess() -> str:
    return (
        "Tai spėjimas, o ne patikrintas faktas: profilis nustatytas pagal "
        "metaduomenis arba failo pavadinimą."
    )


def lut_applied(path: str) -> str:
    return f"Analizei pritaikytas LUT: {path}"


def normalisation_applied() -> str:
    return (
        "Analizei pritaikytas procentilinis kontrasto ir sodrumo normalizavimas. "
        "Tai APYTIKSLIS perskaičiavimas, o ne spalvų valdymo grandinė — "
        "skaičiai nėra kolorimetriškai teisingi."
    )


def no_conversion_applied() -> str:
    return "Analizei jokia spalvų transformacija netaikyta."


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


def reason_sharpness_rank(percentile: float) -> str:
    return f"ryškumas — {percentile:.0f}-as procentilis šiame įraše"


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


COMPONENT_CONTENT = "turinys"
COMPONENT_TECHNICAL = "technika"
COMPONENT_COMPOSITION = "kompozicija"


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

GUI_DROP_HERE = "Nutempkite vaizdo failus čia"
GUI_START = "Pradėti"
GUI_CANCEL = "Atšaukti"
GUI_OPEN_FOLDER = "Atverti rezultatų aplanką"
GUI_IDLE = "Laukiama failų"
GUI_DONE = "Baigta"
GUI_CANCELLED = "Atšaukta"
