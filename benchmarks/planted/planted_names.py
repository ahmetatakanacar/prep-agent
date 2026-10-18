"""Realistic column names for the named variant of each planted dataset.

Every dataset gets a small domain (customers, loans, housing, ...), in Turkish, English or a mix
of both, as tables in this country usually are. The columns are the same as in the anonymous
variant; only the names differ. A leak gets a name that is plausible for a score or a status and
does not announce itself ("risk_skoru", not "leak"), because a name that gives the fault away
would measure name matching instead of judgement.

`numeric` and `categorical` are the names of the base columns in creation order; a recipe uses
the first `n_numeric` and `n_categorical` of them. The generator draws numeric columns from four
shapes in turn (normal around 100, skewed positive, whole numbers from 18 to 79, uniform from 0 to
1000), so the names are ordered to fit: positions 2 and 6 are age-like, 3 is a wide range.
`faults` names the planted columns by fault type (`count_decoy` is the plain count column). A
duplicate is named after its original.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainNames:
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    faults: dict[str, str]


# What is appended to the name of the original to name its copy.
DUPLICATE_SUFFIX = {"duplicate_exact": "_v2", "duplicate_scaled": "_alt"}

NAMES: dict[str, DomainNames] = {
    # telecom customers, Turkish
    "planted-01": DomainNames(
        target="abonelik_iptal",
        numeric=(
            "aylik_ucret",
            "aylik_kullanim_gb",
            "yas",
            "toplam_odeme",
            "arama_sayisi",
            "tekrar_arama_sayisi",
            "abonelik_suresi_ay",
        ),
        categorical=("sehir", "paket_tipi", "odeme_yontemi"),
        faults={
            "leak_linear": "son_donem_risk_skoru",
            "id_sequential": "musteri_no",
            "pii_email": "email",
            "constant": "veri_kaynagi",
            "numeric_category": "tarife_kodu",
        },
    ),
    # consumer loans, English
    "planted-02": DomainNames(
        target="defaulted",
        numeric=(
            "monthly_payment",
            "loan_amount",
            "applicant_age",
            "annual_income",
            "credit_limit",
            "debt_to_income",
            "employment_months",
        ),
        categorical=("employment_type", "loan_purpose", "home_ownership"),
        faults={
            "leak_categorical": "collections_status",
            "id_uuid": "application_uuid",
            "pii_name": "applicant_name",
            "sentinel": "months_since_delinquency",
        },
    ),
    # housing prices, Turkish
    "planted-03": DomainNames(
        target="satis_fiyati",
        numeric=(
            "aidat",
            "merkeze_uzaklik_km",
            "bina_yasi",
            "arsa_alani_m2",
            "gurultu_seviyesi",
            "metroya_mesafe_dk",
            "kiraci_suresi_ay",
        ),
        categorical=("mahalle", "isitma_tipi", "cephe"),
        faults={
            "leak_nonlinear": "ekspertiz_endeksi",
            "id_sequential": "ilan_no",
            "pii_phone": "iletisim_tel",
            "numeric_category": "bina_tipi_kodu",
            "count_decoy": "oda_sayisi",
        },
    ),
    # car insurance, English
    "planted-04": DomainNames(
        target="claim_filed",
        numeric=(
            "premium_amount",
            "annual_mileage",
            "driver_age",
            "vehicle_value",
            "horsepower",
            "claims_history_cost",
            "driver_tenure_months",
        ),
        categorical=("vehicle_type", "region"),
        faults={
            "leak_nonlinear": "claim_risk_index",
            "id_uuid": "policy_guid",
            "pii_email": "contact_email",
            "pii_name": "holder_name",
            "constant": "product_line",
        },
    ),
    # retail stores, Turkish and English mixed
    "planted-05": DomainNames(
        target="aylik_ciro",
        numeric=(
            "avg_basket_value",
            "promosyon_butcesi",
            "magaza_yasi_ay",
            "magaza_alani_m2",
            "kira_bedeli",
            "musteri_trafigi",
            "calisan_sayisi",
        ),
        categorical=("bolge", "store_format", "kategori_tipi"),
        faults={
            "leak_linear": "tahmini_net_gelir",
            "leak_categorical": "ciro_segmenti",
            "pii_phone": "yetkili_gsm",
            "sentinel": "son_denetim_gun",
            "numeric_category": "magaza_tip_kodu",
        },
    ),
    # staff turnover, Turkish and English mixed
    "planted-06": DomainNames(
        target="isten_ayrildi",
        numeric=(
            "performance_score",
            "fazla_mesai_saat",
            "yas",
            "aylik_maas",
            "calisma_saati_ay",
            "egitim_saati",
            "kidem_ay",
        ),
        categorical=("departman", "job_level", "medeni_durum"),
        faults={
            "leak_linear": "memnuniyet_endeksi_son",
            "leak_categorical": "ik_degerlendirme_notu",
            "id_sequential": "sicil_no",
            "pii_phone": "cep_telefonu",
            "numeric_category": "pozisyon_kodu",
            "sentinel": "terfi_sonrasi_gun",
        },
    ),
    # used cars, English with a Turkish seller column
    "planted-07": DomainNames(
        target="sale_price",
        numeric=(
            "horsepower",
            "mileage_1000km",
            "owner_age",
            "listing_views",
            "top_speed_kmh",
            "service_cost",
            "ownership_months",
        ),
        categorical=("brand", "fuel_type"),
        faults={
            "leak_categorical": "price_bracket",
            "leak_nonlinear": "market_value_index",
            "id_uuid": "listing_uuid",
            "pii_name": "satici_ad_soyad",
            "pii_email": "seller_email",
            "constant": "listing_source",
        },
    ),
    # hospital stays, Turkish
    "planted-08": DomainNames(
        target="yeniden_yatis",
        numeric=(
            "sistolik_tansiyon",
            "lab_test_sayisi",
            "hasta_yasi",
            "trigliserid",
            "kan_sekeri",
            "ilac_dozu_mg",
            "yatis_suresi_gun",
        ),
        categorical=("tani_grubu", "sigorta_tipi"),
        faults={
            "id_sequential": "protokol_no",
            "constant": "hastane_kodu",
            "count_decoy": "ameliyat_sayisi",
            "pii_name": "hasta_ad_soyad",
        },
    ),
}
