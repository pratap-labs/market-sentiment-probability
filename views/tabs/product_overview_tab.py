"""Investor-style product overview tab."""

import streamlit as st


def render_product_overview_tab():
    """Render product overview narrative for investors."""
    st.header("✨ Options Income Product Overview")

    st.sidebar.caption("No controls for this tab.")
    st.subheader("What it is")
    st.markdown(
        """
        This product is a disciplined options‑income strategy: it earns premium by selling options on
        index markets and managing the exposure day‑to‑day. The goal is consistent, repeatable income
        rather than betting on big market direction.
        """
    )

    st.subheader("How it makes money")
    st.markdown(
        """
        The engine sells option premium and relies on time decay to harvest income. Each day, it aims
        to capture theta while controlling risk. Premium collected is the primary return driver,
        not prediction of market direction.
        """
    )

    st.subheader("How it controls risk")
    st.markdown(
        """
        - **Directional risk**: caps delta so a one‑sided move doesn’t dominate P&L.
        - **Convexity risk**: manages gamma so near‑expiry moves don’t trigger outsized losses.
        - **Volatility risk**: aligns vega exposure with the volatility regime.
        - **Capital risk**: links position size to NAV and margin so one shock can’t wipe the book.
        """
    )

    st.subheader("Risk analysis & return expectations")
    st.markdown(
        """
        The strategy is designed around **ES99** (Expected Shortfall at 99%). ES99 answers:
        *“In the worst 1% of market outcomes, how much could we lose?”*  
        We use this as the hard risk budget and size positions so ES99 stays within a fixed
        percentage of NAV.

        Returns are a function of how much risk we take. As a rule of thumb, higher premium capture
        requires higher ES99, so we don’t chase yield without expanding the risk budget. In practice,
        the product targets steady monthly income in calm regimes, and scales down exposure when
        volatility rises to preserve capital.
        """
    )

    st.subheader("Second layer: zone‑based testing")
    st.markdown(
        """
        Once ES99 is inside limits, we apply a second control layer: **zone‑based testing**.
        This checks the portfolio’s greeks against predefined safe ranges (Zone 1–3) to ensure
        the book isn’t earning premium by taking hidden convexity or volatility risk.  
        If the portfolio is outside the zone bands, we reduce size or rebalance even if ES99 passes.
        """
    )

    st.subheader("Why it’s a product (not a bet)")
    st.markdown(
        """
        The edge is process: consistent premium capture, systematic risk limits, and rapid adjustments
        when conditions shift. It’s designed to behave like a business with controls, not a single trade.
        """
    )
