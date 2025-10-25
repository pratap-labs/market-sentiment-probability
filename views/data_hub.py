import streamlit as st
import importlib
import os
import sys
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# bring in supabase client to fetch futures data
from database.models import supabase

# Ensure the project root is on sys.path so `from views import ...` works
# This allows running the module as a script (python views/data_hub.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Single place to set page config for combined view
st.set_page_config(layout="wide")

# Import the two view modules lazily
from views import futures_data_loader as futures_view
from views import options_data_loader as options_view


def render():
    st.title("📊 Data Management Hub")
    st.markdown("Use the tabs below to switch between Futures and Options data loaders and visualisers.")

    tab1, tab2, tab3 = st.tabs(["Futures Data", "Options Data", "Analysis"])

    with tab1:
        # Call the futures view renderer
        try:
            futures_view.render()
        except Exception as e:
            st.error(f"Failed to render futures view: {e}")

    with tab2:
        try:
            options_view.render()
        except Exception as e:
            st.error(f"Failed to render options view: {e}")

    with tab3:
        st.subheader("Combined Analysis: Futures & Options per-expiry")

        # Load options data (from files)
        options_df, _ = options_view.load_all_data()

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
                        fig_f.add_trace(go.Scatter(x=subf['date'], y=subf['underlying_value'], name='Underlying', line=dict(color='#9333EA', width=2)), secondary_y=False)
                        fig_f.add_trace(go.Bar(x=subf['date'], y=subf['change_in_oi'], name='Change in OI', marker_color='#3B82F6'), secondary_y=True)
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


if __name__ == "__main__":
    render()
