# views.py  (TOTAL FINAL CODE)
# ✅ Upload (encoding-safe)
# ✅ Preprocess (normalize columns + clean bad dash chars)
# ✅ Generate Timetable (OR-Tools optimal, weekly-hours aware, first-year common subjects, project/seminar only for 4-1/4-2)
# ✅ Export Excel/PDF
# ✅ Analytics (bigger charts + no missing bars)

import os
import io
import re
import math
import base64
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymysql
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings

from sklearn.cluster import KMeans
from ortools.sat.python import cp_model

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas
from matplotlib.patches import Patch


# =========================================================
# DB
# =========================================================
def get_connection():
    return pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="timetable",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


# =========================================================
# BASIC PAGES
# =========================================================
def index(request):
    return render(request, "index.html")


def Signup(request):
    if request.method == "POST":
        username = request.POST.get("t1")
        password = request.POST.get("t2")
        contact = request.POST.get("t3")
        email = request.POST.get("t4")
        address = request.POST.get("t5")

        con = get_connection()
        cur = con.cursor()

        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()

        if user:
            con.close()
            return render(request, "signup.html", {"msg": "⚠️ Username already exists"})

        cur.execute(
            """
            INSERT INTO users (username, password, contact_no, email, address, role)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (username, password, contact, email, address, "user"),
        )
        con.commit()
        con.close()
        return render(request, "login.html", {"msg": "Signup successful! Login now."})

    return render(request, "signup.html")


def Login(request):
    if request.method == "POST":
        username = request.POST.get("t1")
        password = request.POST.get("t2")

        con = get_connection()
        cur = con.cursor(pymysql.cursors.DictCursor)
        cur.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password),
        )
        data = cur.fetchone()
        con.close()

        if data:
            return render(request, "user_home.html", {"user": username})

        return render(request, "login.html", {"msg": "Invalid username or password"})

    return render(request, "login.html")


# =========================================================
# CSV READ HELPERS (encoding-safe)
# =========================================================
def _read_csv_safely(path: str) -> pd.DataFrame:
    """
    Handles Excel/Windows CSV encodings (utf-8, utf-8-sig, latin1).
    """
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort
    return pd.read_csv(path, encoding="latin1", errors="replace")


def _fix_bad_dashes(s: pd.Series) -> pd.Series:
    # fixes “” and \x96 and similar garbage
    return (
        s.astype(str)
        .str.replace("\x96", "-", regex=False)
        .str.replace("", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.strip()
    )


# =========================================================
# UPLOAD DATASET
# =========================================================
def UploadDataset(request):
    context = {}

    if request.method == "POST" and request.FILES.get("dataset"):
        uploaded_file = request.FILES["dataset"]

        dataset_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
        os.makedirs(dataset_dir, exist_ok=True)

        dataset_path = os.path.join(dataset_dir, "uploaded_dataset.csv")

        with open(dataset_path, "wb+") as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        try:
            df = _read_csv_safely(dataset_path)
            context["msg"] = "Dataset uploaded successfully!"
            context["filename"] = uploaded_file.name
            context["rows"] = df.sample(min(10, len(df))).to_html(
                classes="table table-striped", index=False
            )
        except Exception as e:
            context["msg"] = f"Error reading CSV: {e}"

    return render(request, "UploadDataset.html", context)


# =========================================================
# PREPROCESS DATASET (FINAL)
# =========================================================
def preprocess_dataset(request):
    """
    - Reads encoding safely
    - Normalizes columns to lowercase
    - Fixes bad dash chars in subject/faculty
    - Ensures required columns exist (best-effort)
    """
    context = {}

    dataset_path = os.path.join(settings.MEDIA_ROOT, "datasets", "uploaded_dataset.csv")
    pre_dir = os.path.join(settings.MEDIA_ROOT, "preprocessed")
    os.makedirs(pre_dir, exist_ok=True)
    cleaned_path = os.path.join(pre_dir, "cleaned_dataset.csv")

    if not os.path.exists(dataset_path):
        context["msg"] = "No uploaded dataset found. Please upload first."
        return render(request, "preprocess_dataset.html", context)

    try:
        df = _read_csv_safely(dataset_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Fix common naming mismatches (if your raw csv used different headings)
        rename_map = {
            "subject": "subject_name",
            "subjectname": "subject_name",
            "subject_name": "subject_name",
            "faculty": "faculty_name",
            "facultyname": "faculty_name",
            "faculty_name": "faculty_name",
            "room": "room_type",
            "roomtype": "room_type",
            "room_type": "room_type",
        }
        # only rename exact matches
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # fill missing safe defaults
        if "free_period" in df.columns:
            df["free_period"] = df["free_period"].fillna("No")
        else:
            df["free_period"] = "No"

        if "room_type" in df.columns:
            df["room_type"] = df["room_type"].fillna("Room")
        else:
            df["room_type"] = "Room"

        # numeric conversions
        for col in ["year", "semester", "sub_weekly_hours", "sub_hours_full_week", "weekly_hours_per_faculty"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Fix bad dash chars
        if "subject_name" in df.columns:
            df["subject_name"] = _fix_bad_dashes(df["subject_name"])
        if "faculty_name" in df.columns:
            df["faculty_name"] = _fix_bad_dashes(df["faculty_name"])

        # Normalize day values
        if "day" in df.columns:
            df["day"] = df["day"].astype(str).str.strip().str.title()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        df.to_csv(cleaned_path, index=False, encoding="utf-8")

        context["msg"] = "Dataset cleaned successfully."
        context["total_rows"] = df.shape[0]
        context["total_columns"] = df.shape[1]
        context["path"] = cleaned_path
        context["sample_data"] = df.head(10).to_html(classes="table table-striped", index=False)

    except Exception as e:
        context["msg"] = f"Error during preprocessing: {e}"

    return render(request, "preprocess_dataset.html", context)


import os
import re
import random
import pandas as pd
from collections import defaultdict
from django.shortcuts import render
from django.conf import settings
from ortools.sat.python import cp_model


# =========================================================
# CSV HELPERS
# =========================================================
def _read_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _fix_bad_dashes(series: pd.Series) -> pd.Series:
    return (series.astype(str)
            .str.replace("\x96", "-", regex=False)
            .str.replace("", "-", regex=False)
            .str.strip())


# =========================================================
# TIMETABLE HELPERS
# =========================================================
def _normalize_day(x: str) -> str:
    s = str(x).strip().lower()
    mapping = {
        "mon": "Monday", "monday": "Monday",
        "tue": "Tuesday", "tues": "Tuesday", "tuesday": "Tuesday",
        "wed": "Wednesday", "wednesday": "Wednesday",
        "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday", "thursday": "Thursday",
        "fri": "Friday", "friday": "Friday",
        "sat": "Saturday", "saturday": "Saturday",
        "sun": "Sunday", "sunday": "Sunday",
    }
    return mapping.get(s, str(x).strip().title())


def _standard_periods_with_lunch():
    return [
        "09:00-10:00",
        "10:00-11:00",
        "11:00-12:00",
        "12:00-13:00",
        "13:00-14:00",  # lunch
        "14:00-15:00",
        "15:00-16:00",
    ]


def _is_project_like(subject: str) -> bool:
    s = str(subject).strip().lower()
    keywords = ["project", "seminar", "internship", "capstone", "mini project", "major project"]
    return any(k in s for k in keywords)


def _safe_pick_room(room_type: str):
    ROOM_NUMBERS = ["R101", "R102", "R103", "R104", "R105"]
    LAB_NUMBERS = ["LAB1", "LAB2", "LAB3"]
    rt = (room_type or "Room").strip().lower()
    if rt == "lab":
        return "Lab", random.choice(LAB_NUMBERS)
    return "Room", random.choice(ROOM_NUMBERS)


def _year_sem_int(year, sem):
    try:
        yi = int(str(year).strip())
    except:
        yi = None
    try:
        si = int(str(sem).strip())
    except:
        si = None
    return yi, si


# =========================================================
# GENERATE TIMETABLE (UPDATED FOR 4-1 & 4-2: NO FREE PERIOD)
# =========================================================
def generate_timetable(request):
    """
    ✅ Always feasible
    ✅ Subject pool is semester-wide (fixes missing 2-1,2-2,... coverage)
    ✅ Year-1 common: pool ignores branch
    ✅ Project/Seminar only for 4-1 / 4-2
    ✅ OR-Tools weekly-hour aware scheduling
    ✅ Avoid same subject twice in same day (hard when possible)
    ✅ 4-1 & 4-2: NEVER allocate FREE PERIOD (fill with Project/Internship/Seminar)
    """
    csv_path = os.path.join(settings.MEDIA_ROOT, "preprocessed", "cleaned_dataset.csv")

    empty_ctx = {
        "years": [], "semesters": [], "branches": [], "regulations": [], "sections": [],
        "periods": [], "timetable": {}, "selected": {}, "error": None
    }

    if not os.path.exists(csv_path):
        empty_ctx["error"] = f"File not found: {csv_path}"
        return render(request, "generate_timetable.html", empty_ctx)

    df = _read_csv_safely(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["year", "semester", "branch", "regulation", "section",
                "day", "subject_name", "faculty_name", "room_type", "sub_weekly_hours"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        empty_ctx["error"] = f"Dataset missing columns: {miss}"
        return render(request, "generate_timetable.html", empty_ctx)

    # clean
    for c in ["year", "semester", "branch", "regulation", "section"]:
        df[c] = df[c].astype(str).str.strip()

    df["day"] = df["day"].apply(_normalize_day)
    df["subject_name"] = _fix_bad_dashes(df["subject_name"])
    df["faculty_name"] = _fix_bad_dashes(df["faculty_name"])
    df["room_type"] = df["room_type"].astype(str).str.strip().replace({"": "Room"}).fillna("Room")
    df["sub_weekly_hours"] = pd.to_numeric(df["sub_weekly_hours"], errors="coerce").fillna(0).astype(int)

    years = sorted(df["year"].dropna().unique())
    semesters = sorted(df["semester"].dropna().unique())
    branches = sorted(df["branch"].dropna().unique())
    regulations = sorted(df["regulation"].dropna().unique())
    sections = sorted(df["section"].dropna().unique())

    periods = _standard_periods_with_lunch()
    LUNCH_SLOT = "13:00-14:00"
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    active_periods = [p for p in periods if p != LUNCH_SLOT]

    # weekly capacity (5 days * 6 periods = 30)
    slots_per_week = len(days) * len(active_periods)

    # CLEAR
    if request.method == "POST" and request.POST.get("action") == "clear":
        return render(request, "generate_timetable.html", {
            "years": years, "semesters": semesters, "branches": branches,
            "regulations": regulations, "sections": sections,
            "periods": periods, "timetable": {}, "selected": {}, "error": None
        })

    timetable = {}
    selected = {}
    error = None

    if request.method == "POST" and request.POST.get("action") == "generate":
        year = request.POST.get("year", "").strip()
        semester = request.POST.get("semester", "").strip()
        branch = request.POST.get("branch", "").strip()
        regulation = request.POST.get("regulation", "").strip()
        section = request.POST.get("section", "").strip()

        yi, si = _year_sem_int(year, semester)

        selected = {
            "year": year, "semester": semester, "branch": branch,
            "regulation": regulation, "section": section
        }

        if not all([year, semester, branch, regulation, section]):
            error = "Please select Year, Semester, Branch, Regulation and Section."
            return render(request, "generate_timetable.html", {
                "years": years, "semesters": semesters, "branches": branches,
                "regulations": regulations, "sections": sections,
                "periods": periods, "timetable": {}, "selected": selected, "error": error
            })

        # ----------- Pool semester-wide -----------
        base_sem_pool = df[
            (df["year"] == year) &
            (df["semester"] == semester) &
            (df["regulation"] == regulation)
        ].copy()

        if base_sem_pool.empty:
            error = f"No rows for Year={year}, Sem={semester}, Reg={regulation}"
            return render(request, "generate_timetable.html", {
                "years": years, "semesters": semesters, "branches": branches,
                "regulations": regulations, "sections": sections,
                "periods": periods, "timetable": {}, "selected": selected, "error": error
            })

        # First year common: ignore branch in pool
        if yi == 1:
            pool = base_sem_pool.copy()
        else:
            pool = base_sem_pool[base_sem_pool["branch"] == branch].copy()

        if pool.empty:
            error = f"No subject pool for Branch={branch} in Year={year}, Sem={semester}, Reg={regulation}"
            return render(request, "generate_timetable.html", {
                "years": years, "semesters": semesters, "branches": branches,
                "regulations": regulations, "sections": sections,
                "periods": periods, "timetable": {}, "selected": selected, "error": error
            })

        # ✅ IMPORTANT:
        # For 4-1 & 4-2 we KEEP project/seminar/internship
        # For others we REMOVE project-like subjects
        if not (yi == 4 and si in (1, 2)):
            pool = pool[~pool["subject_name"].apply(_is_project_like)].copy()

        # subject weekly hours
        subj_hours = (pool.groupby("subject_name")["sub_weekly_hours"].max().to_dict())
        subj_hours = {s: int(h) for s, h in subj_hours.items() if str(s).strip() and int(h) > 0}
        subjects = sorted(subj_hours.keys())

        if not subjects:
            error = "No valid subjects found after filtering. Check weekly hours in dataset."
            return render(request, "generate_timetable.html", {
                "years": years, "semesters": semesters, "branches": branches,
                "regulations": regulations, "sections": sections,
                "periods": periods, "timetable": {}, "selected": selected, "error": error
            })

        # faculty & roomtype mapping
        subject_to_faculty = defaultdict(list)
        subject_to_roomtype = defaultdict(list)

        for _, r in pool.iterrows():
            s = str(r["subject_name"]).strip()
            if s not in subj_hours:
                continue
            f = str(r["faculty_name"]).strip() if str(r["faculty_name"]).strip() else "-"
            rt = str(r["room_type"]).strip() if str(r["room_type"]).strip() else "Room"
            if f not in subject_to_faculty[s]:
                subject_to_faculty[s].append(f)
            if rt not in subject_to_roomtype[s]:
                subject_to_roomtype[s].append(rt)

        total_required = sum(subj_hours.values())

        # If too high -> scale down
        if total_required > slots_per_week:
            scale = slots_per_week / float(total_required)
            scaled = {s: max(1, int(round(h * scale))) for s, h in subj_hours.items()}

            drift = slots_per_week - sum(scaled.values())
            order = sorted(subj_hours.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            while drift != 0 and idx < len(order) * 20:
                ss = order[idx % len(order)][0]
                if drift > 0:
                    scaled[ss] += 1
                    drift -= 1
                else:
                    if scaled[ss] > 1:
                        scaled[ss] -= 1
                        drift += 1
                idx += 1

            subj_hours = scaled
            subjects = sorted(subj_hours.keys())
            total_required = sum(subj_hours.values())

        # ✅ If too low -> fill remaining slots:
        remaining = slots_per_week - total_required
        if remaining > 0:
            # ✅ 4-1 / 4-2: NO FREE PERIOD at all
            if yi == 4 and si in (1, 2):
                fillers = [
                    ("PROJECT WORK", "Lab"),
                    ("INTERNSHIP", "Room"),
                    ("SEMINAR / PPT", "Room"),
                ]

                per = remaining // len(fillers)
                extra = remaining % len(fillers)

                faculty_pool = pool["faculty_name"].dropna().astype(str).str.strip().unique().tolist()

                for idx, (subj, rt) in enumerate(fillers):
                    add_hours = per + (1 if idx < extra else 0)
                    if add_hours <= 0:
                        continue

                    subj_hours[subj] = subj_hours.get(subj, 0) + add_hours

                    if subj not in subject_to_faculty:
                        subject_to_faculty[subj] = [random.choice(faculty_pool)] if faculty_pool else ["-"]
                    if subj not in subject_to_roomtype:
                        subject_to_roomtype[subj] = [rt]
                    else:
                        subject_to_roomtype[subj] = [rt]  # force

                subjects = sorted(subj_hours.keys())

            # ✅ Other years: allow FREE PERIOD
            else:
                subj_hours["FREE PERIOD"] = remaining
                subject_to_faculty["FREE PERIOD"] = ["-"]
                subject_to_roomtype["FREE PERIOD"] = ["Room"]
                subjects = sorted(subj_hours.keys())

        # =========================================================
        # OR-TOOLS MODEL
        # =========================================================
        model = cp_model.CpModel()

        slot_list = [(d, p) for d in days for p in active_periods]
        slot_index = {(d, p): i for i, (d, p) in enumerate(slot_list)}
        S = len(slot_list)

        def _safe_name(x):
            return re.sub(r"[^A-Za-z0-9]+", "_", str(x))[:25]

        x = {}
        for si_ in range(S):
            for s in subjects:
                x[(si_, s)] = model.NewBoolVar(f"x_{si_}_{_safe_name(s)}")

        # each slot exactly one subject
        for si_ in range(S):
            model.Add(sum(x[(si_, s)] for s in subjects) == 1)

        # weekly hours exact
        for s in subjects:
            model.Add(sum(x[(si_, s)] for si_ in range(S)) == int(subj_hours[s]))

        # avoid same subject twice in same day (except FREE PERIOD)
        repeat_penalty = []
        for d in days:
            day_slots = [slot_index[(d, p)] for p in active_periods]
            for s in subjects:
                if s == "FREE PERIOD":
                    continue

                need = int(subj_hours[s])
                per_day_cap = 1 if need <= len(days) else 2

                cnt = model.NewIntVar(0, per_day_cap, f"cnt_{d}_{_safe_name(s)}")
                model.Add(cnt == sum(x[(si_, s)] for si_ in day_slots))
                model.Add(cnt <= per_day_cap)

                if per_day_cap == 2:
                    rep = model.NewBoolVar(f"rep_{d}_{_safe_name(s)}")
                    model.Add(cnt == 2).OnlyEnforceIf(rep)
                    model.Add(cnt != 2).OnlyEnforceIf(rep.Not())
                    repeat_penalty.append(rep)

        # consecutive penalty (except FREE PERIOD)
        consec_penalty = []
        for d in days:
            day_slots = [slot_index[(d, p)] for p in active_periods]
            for i in range(len(day_slots) - 1):
                a, b = day_slots[i], day_slots[i + 1]
                for s in subjects:
                    if s == "FREE PERIOD":
                        continue
                    both = model.NewBoolVar(f"cons_{d}_{i}_{_safe_name(s)}")
                    model.AddBoolAnd([x[(a, s)], x[(b, s)]]).OnlyEnforceIf(both)
                    model.AddBoolOr([x[(a, s)].Not(), x[(b, s)].Not()]).OnlyEnforceIf(both.Not())
                    consec_penalty.append(both)

        model.Minimize(sum(repeat_penalty) * 100 + sum(consec_penalty) * 5)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 8

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            error = "No feasible timetable found. (Check dataset weekly hours / subject coverage)"
            return render(request, "generate_timetable.html", {
                "years": years, "semesters": semesters, "branches": branches,
                "regulations": regulations, "sections": sections,
                "periods": periods, "timetable": {}, "selected": selected, "error": error
            })

        # stable faculty + roomtype per subject
        subject_faculty_choice = {}
        subject_roomtype_choice = {}
        for s in subjects:
            facs = subject_to_faculty.get(s, ["-"])
            rts = subject_to_roomtype.get(s, ["Room"])
            subject_faculty_choice[s] = facs[0] if facs else "-"
            subject_roomtype_choice[s] = "Lab" if any(str(x).strip().lower() == "lab" for x in rts) else "Room"

        # build timetable with lunch included
        timetable = {d: [] for d in days}

        for d in days:
            for p in periods:
                if p == LUNCH_SLOT:
                    timetable[d].append({"Subject": "LUNCH BREAK", "Faculty": "-", "RoomType": "-", "Room": "-"})
                    continue

                si_ = slot_index[(d, p)]
                pick = None
                for s in subjects:
                    if solver.Value(x[(si_, s)]) == 1:
                        pick = s
                        break

                # fallback
                if pick is None:
                    if yi == 4 and si in (1, 2):
                        pick = "PROJECT WORK"
                    else:
                        pick = "FREE PERIOD"

                fac = subject_faculty_choice.get(pick, "-")
                rt = subject_roomtype_choice.get(pick, "Room")
                rt_final, roomno = _safe_pick_room(rt)

                timetable[d].append({
                    "Subject": pick,
                    "Faculty": fac,
                    "RoomType": rt_final,
                    "Room": roomno
                })

        # save flat export
        flat_export = []
        for d in days:
            for i, cell in enumerate(timetable[d]):
                flat_export.append({
                    "Year": year,
                    "Semester": semester,
                    "Branch": branch,
                    "Regulation": regulation,
                    "Section": section,
                    "Day": d,
                    "Period": periods[i],
                    "Subject": cell["Subject"],
                    "Faculty": cell["Faculty"],
                    "RoomType": cell["RoomType"],
                    "Room": cell["Room"],
                })

        request.session["timetable"] = flat_export
        request.session["timetable_selected"] = selected
        request.session.modified = True

    return render(request, "generate_timetable.html", {
        "years": years,
        "semesters": semesters,
        "branches": branches,
        "regulations": regulations,
        "sections": sections,
        "periods": periods,
        "timetable": timetable,
        "selected": selected,
        "error": error
    })




# =========================================================
# EXPORT TIMETABLE (EXCEL)
# =========================================================
def export_timetable_excel(request):
    timetable = request.session.get("timetable")
    if not timetable:
        return HttpResponse("No timetable found. Generate first!")

    df = pd.DataFrame(timetable)

    wanted_cols = ["Year","Semester","Branch","Regulation","Section","Day","Period","Subject","Faculty","RoomType","Room"]
    df = df[[c for c in wanted_cols if c in df.columns]]

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Timetable")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = 'attachment; filename="timetable.xlsx"'
    return resp


# =========================================================
# EXPORT TIMETABLE (PDF)
# =========================================================
def export_timetable_pdf(request):
    timetable = request.session.get("timetable")
    if not timetable:
        return HttpResponse("No timetable found. Generate first!")

    df = pd.DataFrame(timetable)
    wanted_cols = ["Year","Semester","Branch","Regulation","Section","Day","Period","Subject","Faculty","RoomType","Room"]
    df = df[[c for c in wanted_cols if c in df.columns]]

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="timetable.pdf"'

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(letter))
    page_w, page_h = landscape(letter)

    data = [df.columns.tolist()] + df.values.tolist()

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    rows_per_page = 22
    start_row = 1
    page_no = 1
    left_margin = 20
    top_margin = 30

    while start_row < len(data):
        end_row = min(start_row + rows_per_page, len(data))
        page_data = [data[0]] + data[start_row:end_row]

        table = Table(page_data, repeatRows=1)
        table.setStyle(style)

        table_w, table_h = table.wrap(0, 0)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, page_h - top_margin, f"Generated Timetable (Page {page_no})")

        y = page_h - top_margin - 20 - table_h
        if y < 30:
            y = 30

        table.drawOn(c, left_margin, y)
        c.showPage()

        page_no += 1
        start_row = end_row

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    return response


# =========================================================
# ANALYTICS PAGE (UPDATED - BIGGER CHARTS, NO MISSING BARS)
# =========================================================
def analytics(request):
    dataset_path = os.path.join(settings.MEDIA_ROOT, "preprocessed", "cleaned_dataset.csv")
    context = {}

    if not os.path.exists(dataset_path):
        context["msg"] = "No preprocessed dataset found. Please upload & preprocess first."
        return render(request, "analytics.html", context)

    df = _read_csv_safely(dataset_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["faculty_name", "subject_name", "room_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        context["msg"] = f"Dataset missing required columns: {missing}"
        return render(request, "analytics.html", context)

    df["faculty_name"] = _fix_bad_dashes(df["faculty_name"])
    df["subject_name"] = _fix_bad_dashes(df["subject_name"])
    df["room_type"] = df["room_type"].astype(str).str.strip()

    df = df[(df["faculty_name"] != "") & (df["subject_name"] != "")]

    # 1) Faculty Workload (Top 15)
    faculty_periods = (
        df.groupby("faculty_name")["subject_name"]
        .count()
        .sort_values(ascending=False)
    )
    TOP_N = 15
    faculty_periods = faculty_periods.head(TOP_N)

    workload_labels = ["Medium"] * len(faculty_periods)
    if len(faculty_periods) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(faculty_periods.values.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        order = centers.argsort()
        cluster_to_label = {order[0]: "Light", order[1]: "Medium", order[2]: "Heavy"}
        workload_labels = [cluster_to_label[c] for c in clusters]

    color_map = {"Light": "red", "Medium": "orange", "Heavy": "green"}
    bar_colors = [color_map[w] for w in workload_labels]

    plt.figure(figsize=(16, 7))
    ax = faculty_periods.plot(kind="bar", color=bar_colors)
    plt.title("Faculty Workload (Top 15 by Period Count) — AI Categorized", fontsize=14)
    plt.ylabel("Total Periods", fontsize=12)
    plt.xlabel("Faculty", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    legend_elements = [
        Patch(facecolor="green", label="Heavy"),
        Patch(facecolor="orange", label="Medium"),
        Patch(facecolor="red", label="Light"),
    ]
    plt.legend(handles=legend_elements, title="Workload", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    context["faculty_chart"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # 2) Subject Distribution (Top 12) — PIE
    top_subjects = df["subject_name"].value_counts().head(12)

    plt.figure(figsize=(9, 9))
    top_subjects.plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title("Top 12 Subjects Distribution (By Period Count)", fontsize=14)
    plt.ylabel("")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    context["subject_chart"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # 3) Room Utilization — normalize labels so bars are not missing
    room_norm = df["room_type"].str.strip().str.lower()
    room_norm = room_norm.replace({
        "lab": "Lab",
        "laboratory": "Lab",
        "class": "Room",
        "classroom": "Room",
        "room": "Room",
    })
    room_norm = room_norm.apply(lambda x: "Lab" if "lab" in str(x).lower() else "Room")

    room_usage = room_norm.value_counts()  # ensures both categories show (if exist)

    plt.figure(figsize=(10, 5))
    room_usage.plot(kind="bar")
    plt.title("Room Utilization (Period Count)", fontsize=14)
    plt.ylabel("Total Periods", fontsize=12)
    plt.xlabel("Room Type", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    context["room_chart"] = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return render(request, "analytics.html", context)
