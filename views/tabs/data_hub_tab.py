import streamlit as st
import importlib
import os
import sys
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Ensure the project root is on sys.path so `from views import ...` works
# This allows running the module as a script (python views/data_hub.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Single place to set page config for combined view
# Only set page config when running as standalone script, not when imported
if __name__ == "__main__":
    st.set_page_config(layout="wide")

from database.models import supabase
# Import derivatives data tab for data loading
from views.tabs.derivatives_data_tab import load_from_cache


def render_data_hub_tab():
    """
    Data Hub Tab - Legacy view replaced by Derivatives Data tab.
    This tab now focuses on combined analysis.
    """
    
    st.info("💡 For data fetching and management, please use the **Derivatives Data** tab.")

    tab1, tab2, tab3 = st.tabs(["Futures Data", "Options Data", "Analysis"])

    with tab1:
        st.markdown("### Futures Data")
        st.info("Use the **Derivatives Data** tab to fetch and manage futures data.")
        # Show cached data if available
        futures_df = load_from_cache("nifty_futures")
        if not futures_df.empty:
            st.dataframe(futures_df, use_container_width=True, height=400)

    with tab2:
        st.markdown("### Options Data")
        st.info("Use the **Derivatives Data** tab to fetch and manage options data.")
        # Show cached data if available
        ce_df = load_from_cache("nifty_options_ce")
        pe_df = load_from_cache("nifty_options_pe")
        if not ce_df.empty and not pe_df.empty:
            options_df = pd.concat([ce_df, pe_df], ignore_index=True)
            st.dataframe(options_df, use_container_width=True, height=400)
        elif not ce_df.empty:
            st.dataframe(ce_df, use_container_width=True, height=400)
        elif not pe_df.empty:
            st.dataframe(pe_df, use_container_width=True, height=400)

    with tab3:
        st.subheader("Combined Analysis: Futures & Options per-expiry")

        # Load options data from cache
        ce_df = load_from_cache("nifty_options_ce")
        pe_df = load_from_cache("nifty_options_pe")
        
        if ce_df.empty and pe_df.empty:
            st.warning("⚠️ No options data in cache. Please fetch data from the Derivatives Data tab first.")
            return
        
        options_df = pd.concat([ce_df, pe_df], ignore_index=True) if not ce_df.empty and not pe_df.empty else (ce_df if not ce_df.empty else pe_df)

        # Fetch futures data for the selected symbol
        symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0, key="analysis_symbol")

        # get expiries from both sources (union)
        expiries = set()
        if hasattr(options_df, 'expiry'):
            expiries.update([pd.to_datetime(x) for x in options_df['expiry'].dropna().unique()])

        try:
            fut_res = supabase.table('futures_data')\
                .select('date, expiry_date, underlying_value, open_interest, change_in_oi, symbol')\
                .eq('symbol', symbol)\
                .order('date', desc=False)\
                .execute()
            fut_df = pd.DataFrame(fut_res.data or [])
            if not fut_df.empty and 'expiry_date' in fut_df.columns:
                fut_df['expiry_date'] = pd.to_datetime(fut_df['expiry_date'], errors='coerce')
                expiries.update([pd.to_datetime(x) for x in fut_df['expiry_date'].dropna().unique()])
        except Exception:
            fut_df = pd.DataFrame()

        if not expiries:
            st.info("No expiries found in futures or options data sources.")
            return

        expiries = sorted(list(expiries))
        expiries_str = [d.strftime('%Y-%m-%d') for d in expiries]
        sel = st.multiselect("Select expiries to analyse", expiries_str, default=[expiries_str[-1]])

        for choice in sel:
            expiry_dt = pd.to_datetime(choice)
            st.markdown(f"### Expiry {choice}")

            # ----- Futures plot -----
            # filter futures rows for this expiry
            if fut_df.empty:
                st.info("No futures data available for this symbol")
            else:
                fmask = fut_df['expiry_date'] == expiry_dt
                subf = fut_df.loc[fmask].copy()
                if subf.empty:
                    st.info("No futures rows for this expiry")
                else:
                    subf['date'] = pd.to_datetime(subf['date'], errors='coerce')
                    # last 60 days up to expiry
                    start_dt = expiry_dt - timedelta(days=60)
                    subf = subf[(subf['date'] >= start_dt) & (subf['date'] <= expiry_dt)]
                    if subf.empty:
                        st.info("No futures rows in the last 60 days for this expiry")
                    else:
                        subf = subf.sort_values('date')
                        # coerce numerics
                        subf['underlying_value'] = pd.to_numeric(subf['underlying_value'], errors='coerce')
                        subf['change_in_oi'] = pd.to_numeric(subf['change_in_oi'], errors='coerce').fillna(0)

                        fig_f = make_subplots(specs=[[{'secondary_y': True}]])
                        fig_f.add_trace(go.Scatter(x=subf['date'], y=subf['underlying_value'], name='Underlying', line=dict(color="#F9F6FB", width=2)), secondary_y=False)
                        fig_f.add_trace(go.Bar(x=subf['date'], y=subf['change_in_oi'], name='Change in OI', marker_color="#3B82F6"), secondary_y=True)
                        # tighten layout and remove left/right padding
                        min_dt = subf['date'].min()
                        max_dt = subf['date'].max()
                        pad = pd.Timedelta(hours=12)
                        fig_f.update_layout(title_text=f"Futures — Expiry {choice}", height=420, bargap=0.12, margin=dict(l=40, r=20, t=40, b=60))
                        fig_f.update_xaxes(range=[min_dt - pad, max_dt + pad], tickangle=45, nticks=12)
                        fig_f.update_yaxes(title_text='Underlying', secondary_y=False)
                        fig_f.update_yaxes(title_text='Change in OI', secondary_y=True)
                        st.plotly_chart(fig_f, use_container_width=True)

            # ----- Options plot (render under the futures plot) -----
            if options_df.empty:
                st.info("No options files loaded")
            else:
                opt_sub = options_df[options_df['expiry'] == expiry_dt].copy()
                if opt_sub.empty:
                    st.info("No options rows for this expiry")
                else:
                    opt_sub['date'] = pd.to_datetime(opt_sub['date'], errors='coerce')
                    start_dt = expiry_dt - timedelta(days=60)
                    opt_sub = opt_sub[(opt_sub['date'] >= start_dt) & (opt_sub['date'] <= expiry_dt)]
                    if opt_sub.empty:
                        st.info("No options rows in the last 60 days for this expiry")
                    else:
                        opt_sub['option_type'] = opt_sub['option_type'].astype(str).str.upper().str.strip()
                        oi_pivot = opt_sub.pivot_table(index='date', columns='option_type', values='open_int', aggfunc='sum').fillna(0)
                        for colname in ['CE', 'PE']:
                            if colname not in oi_pivot.columns:
                                oi_pivot[colname] = 0
                        oi_pivot = oi_pivot.sort_index()
                        if 'underlying_value' in opt_sub.columns:
                            underlying = opt_sub.groupby('date')['underlying_value'].last().reindex(oi_pivot.index).ffill()
                        else:
                            underlying = pd.Series([pd.NA] * len(oi_pivot), index=oi_pivot.index)

                        # coerce numerics
                        oi_pivot['CE'] = pd.to_numeric(oi_pivot['CE'], errors='coerce').fillna(0)
                        oi_pivot['PE'] = pd.to_numeric(oi_pivot['PE'], errors='coerce').fillna(0)
                        underlying = pd.to_numeric(underlying, errors='coerce')

                        fig_o = make_subplots(specs=[[{'secondary_y': True}]])
                        fig_o.add_trace(go.Bar(x=oi_pivot.index, y=oi_pivot['CE'], name='CE OI', marker_color='#E96767'), secondary_y=False)
                        fig_o.add_trace(go.Bar(x=oi_pivot.index, y=oi_pivot['PE'], name='PE OI', marker_color='#75F37B'), secondary_y=False)
                        fig_o.add_trace(go.Scatter(x=underlying.index, y=underlying.values, name='Underlying', mode='lines+markers', line=dict(color="#e8efe8", width=2)), secondary_y=True)
                        # remove horizontal padding and tighten bars
                        if len(oi_pivot.index):
                            min_dt = oi_pivot.index.min()
                            max_dt = oi_pivot.index.max()
                        else:
                            min_dt = None
                            max_dt = None
                        pad = pd.Timedelta(hours=12)
                        fig_o.update_layout(title_text=f"Options — Expiry {choice}", barmode='group', height=420, bargap=0.10, margin=dict(l=40, r=20, t=40, b=60))
                        if min_dt is not None and max_dt is not None:
                            fig_o.update_xaxes(range=[min_dt - pad, max_dt + pad], tickangle=45, nticks=12)
                        else:
                            fig_o.update_xaxes(tickangle=45, nticks=12)
                        fig_o.update_yaxes(title_text='Open Interest', secondary_y=False)
                        fig_o.update_yaxes(title_text='Underlying Price', secondary_y=True)
                        st.plotly_chart(fig_o, use_container_width=True)

                        # ----- Per-strike OI chart for a selected day -----
                        # Prepare data: ensure strike_price numeric and date present
                        if 'strike_price' in opt_sub.columns and 'open_int' in opt_sub.columns:
                            opt_sub['strike_price'] = pd.to_numeric(opt_sub['strike_price'], errors='coerce')
                            # Convert available dates to native python datetimes for Streamlit slider
                            available_dates = sorted([pd.to_datetime(d).to_pydatetime() for d in opt_sub['date'].dropna().unique()])
                            if available_dates:
                                # slider for selecting a date (use string representation keys to avoid Streamlit state collisions)
                                slider_key = f"strike_slider_{choice}"
                                # Replace slider with previous/next buttons and weekday display
                                idx_key = f"strike_idx_{choice}"
                                # initialize index to last available date
                                if idx_key not in st.session_state:
                                    st.session_state[idx_key] = len(available_dates) - 1

                                col_prev, col_date, col_next = st.columns([1, 6, 1])
                                if col_prev.button("◀", key=f"prev_{idx_key}"):
                                    st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
                                if col_next.button("▶", key=f"next_{idx_key}"):
                                    st.session_state[idx_key] = min(len(available_dates) - 1, st.session_state[idx_key] + 1)

                                sel_date = available_dates[st.session_state[idx_key]]
                                # show YYYY-MM-DD and day of week (e.g., Fri)
                                col_date.markdown(f"**{sel_date.strftime('%Y-%m-%d (%a)')}**")

                                # convert back to pandas Timestamp for filtering
                                sel_pd = pd.to_datetime(sel_date)
                                day_df = opt_sub[opt_sub['date'] == sel_pd]
                                if day_df.empty:
                                    st.info("No option rows for the selected date at this expiry")
                                else:
                                    # pivot by strike and option_type
                                    day_df['option_type'] = day_df['option_type'].astype(str).str.upper().str.strip()
                                    pivot = day_df.pivot_table(index='strike_price', columns='option_type', values='open_int', aggfunc='sum').fillna(0)
                                    # ensure CE/PE columns
                                    for colname in ['CE', 'PE']:
                                        if colname not in pivot.columns:
                                            pivot[colname] = 0
                                    pivot = pivot.sort_index()
                                    # limit strikes to requested window (23k - 27k)
                                    try:
                                        pivot = pivot[(pivot.index >= 23000) & (pivot.index <= 27000)]
                                    except Exception:
                                        # if index is not numeric, attempt to coerce
                                        pivot.index = pd.to_numeric(pivot.index, errors='coerce')
                                        pivot = pivot.dropna().sort_index()
                                        pivot = pivot[(pivot.index >= 23000) & (pivot.index <= 27000)]

                                    # small bar chart: x=strike, grouped CE/PE
                                    fig_s = make_subplots()
                                    fig_s.add_trace(go.Bar(x=pivot.index, y=pivot['CE'], name='CE OI', marker_color='#E96767'))
                                    fig_s.add_trace(go.Bar(x=pivot.index, y=pivot['PE'], name='PE OI', marker_color='#75F37B'))
                                    fig_s.update_layout(title_text=f"Per-strike OI on {pd.to_datetime(sel_date).strftime('%Y-%m-%d')}", barmode='group', height=420, margin=dict(l=40, r=20, t=40, b=60), bargroupgap=0.02)
                                    fig_s.update_xaxes(title_text='Strike Price')
                                    fig_s.update_yaxes(title_text='Open Interest')
                                    st.plotly_chart(fig_s, use_container_width=True)
                            else:
                                st.info("No dated option rows to build per-strike chart")
