# routers/telemetry.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import json
import fastf1.plotting
import plotly.graph_objects as go
from ..services import FastF1Service
import pandas as pd

router = APIRouter(prefix="/api/telemetry", tags=["Telemetry"])
f1_service = FastF1Service()

# Session name map — frontend sends "Race", FastF1 wants "R"
SESSION_MAP = {
    "Practice 1":          "FP1",
    "Practice 2":          "FP2",
    "Practice 3":          "FP3",
    "Qualifying":          "Q",
    "Race":                "R",
    "Sprint":              "S",
    "Sprint Qualifying":   "SQ",
    "Sprint Shootout":     "SS",
}

@router.get("/position-changes/{year}/{gp}")
async def position_changes(
    year: int,
    gp: str,
    session: str = Query(default="R"),
    drivers: Optional[str] = Query(default=None)
):
    try:
        session_key = SESSION_MAP.get(session, session)
        is_race = session_key in ["R", "S"]

        sess = await f1_service.get_session(year, gp, session_key)
        laps = await f1_service.get_race_laps(year, gp, session_key)
        
        # For non-race sessions, we only want accurate laps for the lap time plot
        if not is_race:
            laps = laps[laps["IsAccurate"] == True]

        # Filter to selected drivers
        if drivers:
            driver_list = [d.strip() for d in drivers.split(',')]
            laps = laps[laps['Driver'].isin(driver_list)]

        if laps.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No lap data found for drivers: {drivers}"
            )

        fig1 = go.Figure()

        # Safety car shading — race/sprint only
        if is_race:
            sc_laps = (
                laps[laps["TrackStatus"].str.contains("4", na=False)]["LapNumber"]
                .unique()
            )
            for lap in sc_laps:
                fig1.add_vrect(
                    x0=lap - 0.5, x1=lap + 0.5,
                    fillcolor="rgba(24, 95, 165, 0.12)",
                    layer="below", line_width=0
                )

        # Driver lines
        teams_seen = set()
        for abb in laps['Driver'].unique():
            drv_laps = laps[laps["Driver"] == abb]
            if drv_laps.empty:
                continue

            try:
                # Use FastF1's built-in style or determine manually
                style = fastf1.plotting.get_driver_style(
                    identifier=abb,
                    style=["color", "linestyle"],
                    session=sess
                )
                colour = style["color"]
                
                # Logic for dashed lines: track team names
                drv_info = sess.get_driver(abb)
                team_name = drv_info["TeamName"]
                
                line_dash = "solid"
                if team_name in teams_seen:
                    line_dash = "dash"
                else:
                    teams_seen.add(team_name)
                    
            except Exception:
                colour = "#888888"
                line_dash = "solid"

            y_values = drv_laps["Position"] if is_race else drv_laps["LapTimeSeconds"]
            
            fig1.add_trace(go.Scatter(
                x=drv_laps["LapNumber"].tolist(),
                y=y_values.tolist(),
                mode="lines+markers" if not is_race else "lines",
                name=abb,
                line=dict(color=colour, width=2, dash=line_dash),
                hovertemplate=(
                    f"<b>{abb}</b><br>"
                    "Lap %{x}<br>"
                    + ("P%{y}" if is_race else "%{y:.3f}s") +
                    "<extra></extra>"
                )
            ))

            # Pit stop markers — race only, INSIDE the driver loop
            if is_race:
                pit_laps = drv_laps[drv_laps["PitInTime"].notna()]
                if not pit_laps.empty:
                    fig1.add_trace(go.Scatter(
                        x=pit_laps["LapNumber"].tolist(),
                        y=pit_laps["Position"].tolist(),
                        mode="markers",
                        name=f"{abb} pit",
                        marker=dict(
                            symbol="diamond", size=8,
                            color=colour,
                            line=dict(color="white", width=1)
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{abb} pit</b><br>"
                            "Lap %{x}<extra></extra>"
                        )
                    ))

        # Layout — OUTSIDE the driver loop, runs for ALL session types
        if not laps.empty:
            max_lap = laps["LapNumber"].max()
            total_laps = int(max_lap) if pd.notna(max_lap) else 1
        else:
            total_laps = 1

        title_suffix = "Position Changes" if is_race else "Lap Times"
        yaxis_title = "Position" if is_race else "Lap Time (seconds)"
        
        yaxis_config = dict(
            title=yaxis_title,
            autorange="reversed" if is_race else True,
        )
        
        if is_race:
            yaxis_config.update(dict(
                range=[20.5, 0.5],
                tickvals=[1, 5, 10, 15, 20],
                ticktext=["P1", "P5", "P10", "P15", "P20"],
            ))

        fig1.update_layout(
            title=f"{gp} {year} — {session} · {title_suffix}",
            xaxis=dict(title="Lap", range=[0, total_laps + 1], dtick=5),
            yaxis=yaxis_config,
            hovermode="closest",
            height=500,
            legend=dict(x=1.02, y=1, font=dict(size=11)),
            margin=dict(l=60, r=120, t=60, b=60),
        )

        return json.loads(fig1.to_json())

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR1: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/tyre-strategy/{year}/{gp}")
async def get_tyre_strategy(year: int, gp: str, session: str = "R", drivers: str = None):
    try:
        session_key = SESSION_MAP.get(session, session)
        sess = await f1_service.get_session(year, gp, session_key)
        
        # Build stints DataFrame
        stints = await f1_service.get_stints(year, gp, session_key)

        # If drivers are passed as comma-separated string, split them
        driver_list = [d.strip() for d in drivers.split(",")] if drivers else stints["Driver"].unique().tolist()
        
        # Filter stints to selected drivers
        stints = stints[stints["Driver"].isin(driver_list)]

        fig = go.Figure()

        # Compounds color map
        COMPOUND_COLORS = {
            "SOFT": "#ff3333",
            "MEDIUM": "#ffff33",
            "HARD": "#f0f0f0",
            "INTERMEDIATE": "#33ff33",
            "WET": "#3333ff",
            "UNKNOWN": "#888888"
        }

        # Build horizontal bar chart per driver
        # We'll group by compound and add traces to allow for a legend (optional)
        for compound in stints["Compound"].unique():
            cp_stints = stints[stints["Compound"] == compound]
            
            # Use FastF1 color if possible
            try:
                color = fastf1.plotting.get_compound_color(compound, session=sess)
            except:
                color = COMPOUND_COLORS.get(compound.upper(), COMPOUND_COLORS["UNKNOWN"])

            fig.add_trace(go.Bar(
                x=cp_stints["StintLength"],
                y=cp_stints["Driver"],
                orientation='h',
                name=compound,
                marker=dict(color=color),
                hovertemplate="Driver: %{y}<br>Compound: " + compound + "<br>Laps: %{x}<extra></extra>"
            ))

        # Layout styling
        fig.update_layout(
            title=f"{year} {gp} Tyre Strategies — {session}",
            xaxis_title="Lap Number",
            yaxis=dict(autorange="reversed"),
            barmode='stack',
            height=200 + (len(driver_list) * 25), # Dynamic height
            margin=dict(l=60, r=20, t=60, b=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb")
        )

        return json.loads(fig.to_json())
    except Exception as e:
        print(f"ERROR2: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gear-shifts/{year}/{gp}")
async def get_gear_shifts(year: int, gp: str, session: str = "R", drivers: str = None):
    """Return precomputed fastest-lap gear traces for all requested drivers."""
    try:
        session_key = SESSION_MAP.get(session, session)
        sess = await f1_service.get_session(year, gp, session_key)
        driver_list = [d.strip() for d in drivers.split(",")] if drivers else list(sess.drivers)

        fig = go.Figure()
        available_drivers = []
        gear_colors = {
            1: "#6b7280", 2: "#60a5fa", 3: "#34d399", 4: "#facc15",
            5: "#fb923c", 6: "#f87171", 7: "#c084fc", 8: "#f472b6",
        }

        for driver in driver_list:
            laps = sess.laps.pick_drivers(driver).pick_fastest()
            if laps.empty:
                continue

            telemetry = laps.get_telemetry()
            if telemetry.empty or not {"X", "Y", "nGear"}.issubset(telemetry.columns):
                continue

            telemetry = telemetry.dropna(subset=["X", "Y", "nGear"])
            if telemetry.empty:
                continue

            driver_info = sess.get_driver(driver)
            abbreviation = driver_info["Abbreviation"]
            available_drivers.append(abbreviation)

            try:
                colour = "#" + driver_info["TeamColor"]
            except (KeyError, TypeError):
                colour = "#ef4444"

            fig.add_trace(go.Scatter(
                x=telemetry["X"].tolist(),
                y=telemetry["Y"].tolist(),
                mode="lines",
                name=f"{abbreviation} track",
                legendgroup=abbreviation,
                line=dict(color=colour, width=2),
                meta={"driver": abbreviation, "kind": "track"},
                hoverinfo="skip",
                showlegend=False,
            ))

            for gear, gear_data in telemetry.groupby(telemetry["nGear"].round().astype(int)):
                fig.add_trace(go.Scatter(
                    x=gear_data["X"].tolist(),
                    y=gear_data["Y"].tolist(),
                    mode="markers",
                    name=f"{abbreviation} gear {gear}",
                    legendgroup=abbreviation,
                    marker=dict(color=gear_colors.get(gear, "#ffffff"), size=5),
                    meta={"driver": abbreviation, "kind": "gear", "gear": gear},
                    hovertemplate=f"<b>{abbreviation}</b><br>Gear {gear}<extra></extra>",
                    showlegend=True,
                ))

        if not available_drivers:
            raise HTTPException(status_code=404, detail="No telemetry gear data found for the selected drivers")

        fig.update_layout(
            title=f"{year} {gp} — {session} Gear Shifts",
            xaxis=dict(title="Track X", visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Track Y", visible=False),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(title="Gear", orientation="h", y=-0.08),
            hovermode="closest",
        )
        fig.update_layout(meta={"drivers": available_drivers})
        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR GEAR SHIFTS: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speed-traces/{year}/{gp}")
async def get_speed_traces(year: int, gp: str, session: str = "R", drivers: str = None):
    """Return fastest-lap speed traces and shared circuit corner metadata."""
    try:
        session_key = SESSION_MAP.get(session, session)
        sess = await f1_service.get_session(year, gp, session_key)
        driver_list = [d.strip() for d in drivers.split(",")] if drivers else list(sess.drivers)

        fig = go.Figure()
        available_drivers = []
        reference_telemetry = None

        for driver in driver_list:
            lap = sess.laps.pick_drivers(driver).pick_fastest()
            if lap.empty:
                continue

            telemetry = lap.get_telemetry()
            required_columns = {"Distance", "Speed"}
            if telemetry.empty or not required_columns.issubset(telemetry.columns):
                continue

            telemetry = telemetry.dropna(subset=list(required_columns))
            if telemetry.empty:
                continue

            driver_info = sess.get_driver(driver)
            abbreviation = driver_info["Abbreviation"]
            available_drivers.append(abbreviation)
            if reference_telemetry is None:
                reference_telemetry = telemetry

            try:
                colour = "#" + driver_info["TeamColor"]
            except (KeyError, TypeError):
                colour = "#ef4444"

            fig.add_trace(go.Scatter(
                x=telemetry["Distance"].round(2).tolist(),
                y=telemetry["Speed"].round(2).tolist(),
                mode="lines",
                name=abbreviation,
                line=dict(color=colour, width=2),
                meta={"driver": abbreviation, "kind": "speed"},
                hovertemplate=(
                    f"<b>{abbreviation}</b><br>"
                    "Distance: %{x:.0f} m<br>Speed: %{y:.0f} km/h<extra></extra>"
                ),
            ))

        if not available_drivers or reference_telemetry is None:
            raise HTTPException(status_code=404, detail="No speed telemetry found for the selected drivers")

        # Circuit annotations are supplemental FastF1 data and are not
        # available for every event/session, especially newly added events.
        try:
            circuit_info = sess.get_circuit_info()
            corners = circuit_info.corners.copy() if circuit_info is not None else pd.DataFrame()
        except (AttributeError, KeyError, TypeError) as exc:
            print(f"[SPEED TRACES] Corner metadata unavailable: {exc}")
            corners = pd.DataFrame()
        corner_x = []
        corner_y = []
        corner_text = []
        corner_distances = []
        max_speed = float(reference_telemetry["Speed"].max())

        for _, corner in corners.iterrows():
            if not pd.notna(corner.get("X")) or not pd.notna(corner.get("Y")):
                continue

            corner_position = ((
                reference_telemetry["X"] - float(corner["X"])
            ) ** 2 + (
                reference_telemetry["Y"] - float(corner["Y"])
            ) ** 2).idxmin()
            distance = float(reference_telemetry.loc[corner_position, "Distance"])
            label = f"{corner['Number']}{corner.get('Letter', '')}"
            corner_distances.append(distance)
            corner_x.append(distance)
            corner_y.append(max_speed + 4)
            corner_text.append(label)

        for distance, label in zip(corner_distances, corner_text):
            fig.add_vline(
                x=distance,
                line_width=1,
                line_dash="dot",
                line_color="rgba(156, 163, 175, 0.45)",
                layer="below",
            )

        fig.add_trace(go.Scatter(
            x=corner_x,
            y=corner_y,
            mode="markers+text",
            text=corner_text,
            textposition="top center",
            name="Corners",
            marker=dict(color="#f9fafb", size=7, symbol="diamond"),
            meta={"kind": "corners"},
            hovertemplate="Corner %{text}<br>Distance: %{x:.0f} m<extra></extra>",
            showlegend=True,
        ))

        fig.update_layout(
            title=f"{year} {gp} — {session} Speed Traces",
            xaxis=dict(title="Distance (m)", rangemode="tozero"),
            yaxis=dict(title="Speed (km/h)"),
            height=500,
            margin=dict(l=60, r=30, t=70, b=60),
            hovermode="x unified",
            legend=dict(title="Driver", bgcolor="rgba(0,0,0,0)"),
            meta={"drivers": available_drivers, "corners": corner_text},
        )
        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR SPEED TRACES: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
# sample
# @router.get("/tyre-strategy/{year}/{gp}")
# async def get_tyre_strategy(year: int, gp: str, session: str = "R", drivers: str = None):
#     try:
#         session_key = SESSION_MAP.get(session, session)
#         is_race = session_key in ["R", "S"]

#         sess = await f1_service.get_session(year, gp, session_key)
#         laps = await f1_service.get_race_laps(year, gp, session_key)
#         fig = go.Figure()

#         # to do
        

#         return json.loads(fig.to_json())
#     except Exception as e:
#         print(f"ERROR2: {type(e).__name__}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))
    