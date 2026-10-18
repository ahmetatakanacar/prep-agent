"""The recipes of set D: twelve datasets, six for development and six for one final test.

Set D is independent of the first three sets. Its domains, column names, column counts and row
counts are new, and what it asks is different: it does not only plant faults that give
themselves away, it plants columns whose association with the target is partial or very strong
and says, by what the column means, whether it is a leak or a legitimate feature.

- `leak_partial`: information that only exists after the outcome (a status written afterwards, a
  cost computed from the result, a field filled in when the event happened). Must be dropped.
- `legit_strong`: a real property that is known before the outcome and predicts it well (the odor
  of a mushroom, the floor area of a building). Must be kept, even when it is as strong as a leak.

Both come with a `level`, the association with the target they were made to have: 0.6, 0.75,
0.85 and 0.95 for leaks, and 0.6 to 0.97 for legitimate features. At every level there are
columns of both kinds, because from the numbers alone they cannot be told apart. The other fault
types are in the set too, in less obvious forms (see `planted_generate_d`).

The `dev` recipes are for looking at results. The `test` recipes are run once, at the end, and
their data and results are not looked at before that: only their recipes and ground truth are in
the repository.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Col:
    """A base column that feeds the target. `loc` and `scale` shape its distribution (for
    `normal` the mean and standard deviation, for `lognormal` those of the logarithm, for `age`
    the first and last whole number, for `uniform` the range, for `count` the mean)."""

    name: str
    kind: Literal["normal", "lognormal", "age", "uniform", "count", "cat"]
    loc: float | None = None
    scale: float | None = None
    levels: int = 0
    missing: float = 0.0


@dataclass(frozen=True)
class Fault:
    fault: str
    name: str
    level: float | None = None
    form: str = "default"
    # The name (in the named variant) of the base column a duplicate copies.
    of: str | None = None
    # For a scaled duplicate: the factor of the unit conversion.
    factor: float | None = None
    loc: float = 50.0
    scale: float = 10.0
    reason: str = ""


@dataclass(frozen=True)
class RecipeD:
    name: str
    seed: int
    split: Literal["dev", "test"]
    task: Literal["classification", "regression"]
    n_rows: int
    target: str
    base: tuple[Col, ...]
    faults: tuple[Fault, ...] = field(default_factory=tuple)
    pos_rate: float = 0.4
    mu: float = 0.0
    sigma: float = 1.0


def leak(name, level, form, reason, **kw):
    return Fault("leak_partial", name, level=level, form=form, reason=reason, **kw)


def legit(name, level, form, reason, **kw):
    return Fault("legit_strong", name, level=level, form=form, reason=reason, **kw)


DEV: tuple[RecipeD, ...] = (
    RecipeD(
        "planted-d01",
        4101,
        "dev",
        "classification",
        1800,
        "cancelled",
        base=(
            Col("lead_time_days", "lognormal", 4.0, 0.9),
            Col("nightly_rate", "normal", 140, 30),
            Col("guest_age", "age", 19, 78),
            Col("stay_nights", "count", 3),
            Col("special_requests", "count", 1),
            Col("previous_stays", "count", 1.2, missing=0.06),
            Col("market_segment", "cat", levels=5),
            Col("room_type", "cat", levels=6),
            Col("meal_plan", "cat", levels=3, missing=0.08),
        ),
        faults=(
            legit(
                "deposit_policy",
                0.97,
                "category",
                "the deposit terms are agreed at booking, before any cancellation",
            ),
            leak(
                "reservation_status_note",
                0.85,
                "category",
                "a note the front desk writes after the stay or the cancellation",
            ),
            leak(
                "refund_issued_on",
                0.6,
                "gap",
                "a date that exists only for cancelled bookings that were refunded",
            ),
            Fault("id_sequential", "booking_ref", form="sparse"),
            Fault("pii_email", "guest_email", form="initial"),
            Fault(
                "duplicate_scaled",
                "nightly_rate_gbp",
                form="coarse",
                of="nightly_rate",
                factor=0.79,
            ),
            Fault("numeric_category", "channel_code"),
        ),
        pos_rate=0.37,
    ),
    RecipeD(
        "planted-d02",
        4102,
        "dev",
        "regression",
        1200,
        "annual_kwh",
        base=(
            Col("building_age", "age", 2, 90),
            Col("occupant_density", "normal", 18, 5),
            Col("hvac_runtime_hours", "lognormal", 7.6, 0.25),
            Col("lighting_load_w", "uniform", 2000, 15000),
            Col("window_ratio", "uniform", 0.1, 0.6),
            Col("orientation", "cat", levels=4),
            Col("heating_system", "cat", levels=5),
            Col("roof_type", "cat", levels=3),
            Col("last_retrofit", "age", 1995, 2022, missing=0.31),
            Col("floors", "count", 3),
        ),
        faults=(
            legit(
                "floor_area_m2",
                0.93,
                "numeric",
                "the size of the building is a plan fact, known before any consumption",
                loc=420,
                scale=120,
            ),
            leak(
                "annual_cost_estimate",
                0.95,
                "numeric",
                "the energy bill, which is the consumption times a tariff",
                loc=15000,
                scale=4000,
            ),
            leak(
                "efficiency_band",
                0.75,
                "category",
                "a label an auditor assigns after reading the meter",
            ),
            Fault("id_uuid", "meter_serial", form="hex32"),
            Fault("pii_name", "owner_name", form="surname_first"),
            Fault("constant", "region_code"),
            Fault(
                "duplicate_scaled",
                "occupant_density_per_100m2",
                of="occupant_density",
                factor=100.0,
            ),
        ),
        mu=120000,
        sigma=38000,
    ),
    RecipeD(
        "planted-d03",
        4103,
        "dev",
        "classification",
        1050,
        "gecti",
        base=(
            Col("haftalik_calisma_saati", "lognormal", 2.1, 0.5),
            Col("devam_orani", "normal", 82, 10),
            Col("onceki_not_ortalamasi", "normal", 68, 12),
            Col("odev_teslim_sayisi", "count", 7),
            Col("ogrenci_yasi", "age", 17, 31),
            Col("bolum", "cat", levels=6),
            Col("burs_durumu", "cat", levels=3, missing=0.12),
            Col("ders_yuku", "count", 5),
        ),
        faults=(
            leak(
                "final_notu",
                0.75,
                "numeric",
                "the grade of the exam whose result is the target",
                loc=62,
                scale=14,
            ),
            leak(
                "sertifika_tarihi",
                0.95,
                "gap",
                "the date a certificate is issued, which only passing students get",
            ),
            legit(
                "vize_notu",
                0.85,
                "numeric",
                "the midterm is taken before the final it predicts",
                loc=58,
                scale=15,
            ),
            legit(
                "tekrar_gecmisi",
                0.93,
                "category",
                "how often the student retook courses is history before this exam",
            ),
            Fault("id_sequential", "ogrenci_no", form="prefixed"),
            Fault("pii_phone", "veli_telefonu", form="int10"),
            Fault("sentinel", "son_giris_gun"),
        ),
        pos_rate=0.62,
    ),
    RecipeD(
        "planted-d04",
        4104,
        "dev",
        "regression",
        2200,
        "gecikme_saat",
        base=(
            Col("package_weight_kg", "lognormal", 2.6, 0.9),
            Col("pickup_hour", "age", 6, 20),
            Col("customs_checks", "count", 1),
            Col("fuel_price_index", "normal", 100, 8),
            Col("driver_experience_years", "age", 1, 25),
            Col("vehicle_age_years", "age", 0, 14, missing=0.07),
            Col("carrier_code", "cat", levels=6),
            Col("origin_region", "cat", levels=8),
            Col("destination_region", "cat", levels=8),
            Col("cargo_class", "cat", levels=4),
            Col("weather_flag", "cat", levels=3, missing=0.2),
        ),
        faults=(
            leak(
                "actual_transit_hours",
                0.85,
                "numeric",
                "measured on delivery, so it is the delay plus the planned time",
                loc=48,
                scale=14,
            ),
            leak(
                "delay_reason_code",
                0.6,
                "category",
                "a code the dispatcher enters after a shipment turned out late",
            ),
            legit(
                "route_distance_km",
                0.93,
                "numeric",
                "the planned route is known when the shipment is booked",
                loc=900,
                scale=350,
            ),
            Fault("duplicate_exact", "carrier_name", form="relabel", of="carrier_code"),
            Fault("numeric_category", "service_level_code"),
            Fault("count_decoy", "stops_count"),
            Fault("id_uuid", "consignment_guid", form="standard"),
        ),
        mu=12,
        sigma=8,
    ),
    RecipeD(
        "planted-d05",
        4105,
        "dev",
        "classification",
        1500,
        "disease_present",
        base=(
            Col("leaf_area_cm2", "lognormal", 3.2, 0.5),
            Col("soil_ph", "normal", 6.4, 0.7),
            Col("rainfall_mm_30d", "uniform", 10, 220),
            Col("plant_age_days", "age", 20, 140),
            Col("nitrogen_ppm", "normal", 42, 9),
            Col("field_slope", "uniform", 0, 18),
            Col("crop_variety", "cat", levels=7),
            Col("irrigation", "cat", levels=3),
        ),
        faults=(
            legit(
                "leaf_spot_pattern",
                0.97,
                "category",
                "the pattern of spots is observed in the field, like the odor of a mushroom",
            ),
            legit(
                "humidity_index",
                0.75,
                "numeric",
                "measured by the field sensor before the diagnosis",
                loc=70,
                scale=12,
            ),
            leak(
                "lab_confirmation_score",
                0.95,
                "numeric",
                "the lab's score, which exists only after a sample was sent for the diagnosis",
                loc=40,
                scale=18,
            ),
            leak(
                "treatment_cost",
                0.6,
                "numeric",
                "money spent after the disease was found",
                loc=300,
                scale=90,
            ),
            Fault("constant", "survey_round"),
            Fault("pii_email", "grower_email", form="default"),
            Fault("id_sequential", "plot_ref", form="sparse"),
        ),
        pos_rate=0.45,
    ),
    RecipeD(
        "planted-d06",
        4106,
        "dev",
        "regression",
        950,
        "hasar_tutari",
        base=(
            Col("surucu_yasi", "age", 20, 72),
            Col("ehliyet_yili", "age", 1, 40),
            Col("arac_yasi", "age", 0, 18),
            Col("yillik_km", "lognormal", 9.4, 0.5),
            Col("prim_tutari", "normal", 5200, 900),
            Col("hasarsizlik_kademesi", "count", 3),
            Col("ikamet_ili", "cat", levels=7),
            Col("kullanim_tipi", "cat", levels=3),
        ),
        faults=(
            legit(
                "arac_degeri",
                0.85,
                "numeric",
                "the value of the car is on the policy before any accident",
                loc=450000,
                scale=140000,
            ),
            leak(
                "eksper_rezervi",
                0.75,
                "numeric",
                "the reserve an appraiser sets after inspecting the damage",
                loc=9000,
                scale=3000,
            ),
            leak(
                "odeme_kademesi",
                0.85,
                "category",
                "the settlement tier is decided after the damage is known",
            ),
            legit(
                "kasa_tipi",
                0.6,
                "category",
                "the body type is on the registration, and repair costs differ by it",
            ),
            Fault("id_uuid", "dosya_guid", form="standard"),
            Fault("pii_name", "hak_sahibi", form="initial"),
            Fault(
                "duplicate_scaled", "prim_tutari_usd", form="coarse", of="prim_tutari", factor=0.031
            ),
            Fault("sentinel", "son_hasar_ay"),
        ),
        mu=18000,
        sigma=9000,
    ),
)

TEST: tuple[RecipeD, ...] = (
    RecipeD(
        "planted-t01",
        4201,
        "test",
        "regression",
        1400,
        "resale_value",
        base=(
            Col("battery_health_pct", "normal", 86, 8),
            Col("storage_gb", "count", 3),
            Col("age_months", "age", 3, 54),
            Col("screen_cracks", "count", 0.6),
            Col("seller_rating", "normal", 4.4, 0.4, missing=0.1),
            Col("model_family", "cat", levels=8),
            Col("carrier_lock", "cat", levels=3),
            Col("colour", "cat", levels=6),
        ),
        faults=(
            legit(
                "launch_price",
                0.93,
                "numeric",
                "the price when the phone was released is public history",
                loc=780,
                scale=260,
            ),
            leak(
                "buyer_negotiated_discount",
                0.75,
                "numeric",
                "the discount the buyer got, agreed while the sale closed",
                loc=60,
                scale=25,
            ),
            leak(
                "sold_price_bracket",
                0.95,
                "category",
                "a price bracket assigned from the final sale price",
            ),
            Fault("id_uuid", "handset_guid", form="hex32"),
            Fault("pii_email", "vendor_mail", form="initial"),
            Fault("duplicate_exact", "colour_label", form="relabel", of="colour"),
            Fault("numeric_category", "grade_code"),
        ),
        mu=420,
        sigma=190,
    ),
    RecipeD(
        "planted-t02",
        4202,
        "test",
        "classification",
        2000,
        "yolcu_iptal",
        base=(
            Col("tahmini_sure_dk", "lognormal", 2.6, 0.5),
            Col("mesafe_km", "lognormal", 1.9, 0.7),
            Col("surucu_puani", "normal", 4.6, 0.3),
            Col("yolcu_gecmis_yolculuk", "count", 12, missing=0.04),
            Col("talep_saati", "age", 0, 23),
            Col("talep_semti", "cat", levels=9),
            Col("odeme_tipi", "cat", levels=3),
        ),
        faults=(
            legit(
                "alim_noktasi_tipi",
                0.97,
                "category",
                "the kind of pickup point is chosen in the request, before it is cancelled",
            ),
            legit(
                "tahmini_varis_dk",
                0.85,
                "numeric",
                "the app shows the driver arrival estimate when the ride is requested",
                loc=9,
                scale=4,
            ),
            leak(
                "surucu_iptal_nedeni",
                0.85,
                "gap",
                "a reason the driver app records only when the ride was cancelled",
            ),
            leak(
                "destek_bileti_suresi",
                0.6,
                "numeric",
                "minutes spent on a support ticket opened after the cancellation",
                loc=12,
                scale=6,
            ),
            Fault("id_sequential", "talep_no", form="prefixed"),
            Fault("pii_phone", "yolcu_telefon", form="default"),
            Fault("duplicate_exact", "bolge_adi", form="relabel", of="talep_semti"),
        ),
        pos_rate=0.3,
    ),
    RecipeD(
        "planted-t03",
        4203,
        "test",
        "classification",
        1300,
        "stockout",
        base=(
            Col("weekly_demand", "lognormal", 4.4, 0.6),
            Col("lead_time_days", "age", 2, 30),
            Col("supplier_reliability", "normal", 0.9, 0.06),
            Col("shelf_life_days", "uniform", 10, 400),
            Col("units_on_hand", "lognormal", 4.8, 0.8),
            Col("category", "cat", levels=10),
            Col("storage_zone", "cat", levels=4),
            Col("promo_active", "cat", levels=2, missing=0.05),
        ),
        faults=(
            leak(
                "emergency_order_qty",
                0.95,
                "numeric",
                "units ordered in a rush, which is done only once the shelf is empty",
                loc=120,
                scale=45,
            ),
            leak(
                "restock_priority",
                0.75,
                "category",
                "a priority label the planner sets after a stockout is reported",
            ),
            legit(
                "days_of_cover",
                0.93,
                "numeric",
                "stock divided by demand is known every day, before a stockout",
                loc=14,
                scale=6,
            ),
            legit(
                "reorder_gap",
                0.6,
                "numeric",
                "how far stock is under the reorder point, known in advance",
                loc=30,
                scale=15,
            ),
            Fault("id_uuid", "sku_guid", form="standard"),
            Fault("pii_name", "buyer_contact", form="surname_first"),
            Fault("constant", "warehouse_id"),
            Fault("sentinel", "last_audit_days"),
        ),
        pos_rate=0.4,
    ),
    RecipeD(
        "planted-t04",
        4204,
        "test",
        "regression",
        1250,
        "yillik_uretim_kwh",
        base=(
            Col("panel_sayisi", "age", 6, 60),
            Col("egim_derece", "uniform", 5, 45),
            Col("golgelenme_orani", "uniform", 0, 0.4),
            Col("panel_yasi", "age", 0, 20),
            Col("inverter_verimi", "normal", 0.96, 0.015),
            Col("tesis_ili", "cat", levels=8),
            Col("montaj_tipi", "cat", levels=3),
            Col("yon", "cat", levels=4, missing=0.09),
        ),
        faults=(
            legit(
                "gunes_isinimi",
                0.93,
                "numeric",
                "the average irradiance of the site is a measured property of the place",
                loc=1650,
                scale=240,
            ),
            leak(
                "performans_sinifi",
                0.85,
                "category",
                "a grade given to the plant after a year of metered production",
            ),
            leak(
                "alarm_sayisi",
                0.6,
                "numeric",
                "alarms the monitoring system logged during the year being predicted",
                loc=8,
                scale=5,
            ),
            Fault("id_sequential", "tesis_no", form="sparse"),
            Fault("pii_email", "sahip_eposta", form="default"),
            Fault("duplicate_scaled", "panel_yasi_ay", of="panel_yasi", factor=12.0),
            Fault("count_decoy", "inverter_sayisi"),
        ),
        mu=38000,
        sigma=15000,
    ),
    RecipeD(
        "planted-t05",
        4205,
        "test",
        "classification",
        1100,
        "gec_iade",
        base=(
            Col("uyelik_suresi_yil", "age", 0, 25),
            Col("yillik_odunc_sayisi", "lognormal", 2.4, 0.7),
            Col("kitap_sayfa", "uniform", 60, 900),
            Col("kutuphane_uzaklik_km", "lognormal", 0.8, 0.8),
            Col("yas_grubu", "cat", levels=5),
            Col("sube", "cat", levels=6),
            Col("dil", "cat", levels=4, missing=0.1),
        ),
        faults=(
            leak(
                "ceza_odendi_tarihi",
                0.95,
                "gap",
                "the date a late fee was paid, which exists only for late returns",
            ),
            leak(
                "hatirlatma_epostasi_sayisi",
                0.75,
                "numeric",
                "reminders are sent after the due date has passed",
                loc=2,
                scale=1.5,
            ),
            legit(
                "odunc_sinifi",
                0.97,
                "category",
                "short-loan or standard is a property of the item when it is borrowed",
            ),
            legit(
                "yenileme_sayisi",
                0.85,
                "numeric",
                "renewals are made before the due date",
                loc=1.5,
                scale=1,
            ),
            Fault("id_uuid", "uye_guid", form="hex32"),
            Fault("pii_phone", "uye_telefon", form="int10"),
            Fault("pii_name", "uye_ad_soyad", form="initial"),
            Fault("numeric_category", "kategori_kodu"),
        ),
        pos_rate=0.33,
    ),
    RecipeD(
        "planted-t06",
        4206,
        "test",
        "regression",
        850,
        "monthly_revenue",
        base=(
            Col("opening_years", "age", 0, 30),
            Col("avg_ticket", "normal", 24, 6),
            Col("staff_count", "age", 4, 40),
            Col("rating", "normal", 4.1, 0.4),
            Col("rent_per_month", "lognormal", 8.2, 0.5),
            Col("neighbourhood", "cat", levels=9),
            Col("cuisine", "cat", levels=7),
        ),
        faults=(
            leak(
                "revenue_tier_label",
                0.6,
                "category",
                "a tier the owners association assigns after the year's revenue is reported",
            ),
            leak(
                "delivery_commission_total",
                0.85,
                "numeric",
                "a share of delivery revenue, so it is part of the revenue",
                loc=3200,
                scale=900,
            ),
            legit(
                "seat_count",
                0.75,
                "numeric",
                "the capacity of the room is a fixed fact",
                loc=48,
                scale=16,
            ),
            Fault("id_sequential", "store_ref", form="dense"),
            Fault("pii_email", "owner_email", form="default"),
            Fault("duplicate_scaled", "avg_ticket_eur", of="avg_ticket", factor=0.92),
            Fault("constant", "currency"),
        ),
        mu=52000,
        sigma=21000,
    ),
)

ALL_D: tuple[RecipeD, ...] = DEV + TEST
