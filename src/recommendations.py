"""
recommendations.py
-------------------
Generates actionable business recommendations, marketing strategies,
and targeting guidance for each customer segment.
"""


class RecommendationEngine:
    """Produce segment-level and overall business recommendations."""

    # ------------------------------------------------------------------ #
    #  Per-segment recommendations
    # ------------------------------------------------------------------ #
    def generate_segment_recommendations(self, profiles: list) -> list:
        """
        For each profile (must include 'segment_name'), generate:
          - business_interpretation (2-3 sentences)
          - marketing_strategy      (2-3 bullet points)
          - target_channels         (list of channel names)

        Returns a list of recommendation dicts.
        """
        recommendations = []

        for p in profiles:
            segment = p.get("segment_name", "Unknown")
            seg_lower = segment.lower()

            # --- Determine income and spending level flags ---
            high_income = "high income" in seg_lower
            low_income = "low income" in seg_lower
            high_spending = "high spenders" in seg_lower
            low_spending = "low spenders" in seg_lower
            average = "average" in seg_lower

            # --- Business interpretation ---
            if high_income and high_spending:
                interpretation = (
                    f"Cluster {p['cluster_id']} represents premium customers with both high "
                    f"income (avg ${p['mean_income']}k) and high spending scores "
                    f"(avg {p['mean_spending_score']}). They are the most valuable segment "
                    f"and should be prioritised for loyalty and retention programmes."
                )
                strategy = [
                    "Launch an exclusive VIP loyalty programme with premium rewards.",
                    "Offer early access to new products and personalised concierge services.",
                    "Create invitation-only events to deepen brand affinity.",
                ]
                channels = ["Email (personalised)", "Mobile App push notifications",
                            "In-store personal shoppers", "Exclusive events"]

            elif high_income and low_spending:
                interpretation = (
                    f"Cluster {p['cluster_id']} earns well (avg ${p['mean_income']}k) but "
                    f"spends conservatively (avg score {p['mean_spending_score']}). "
                    f"They have untapped purchasing power — the goal is to increase their "
                    f"engagement and spending through targeted promotions."
                )
                strategy = [
                    "Send targeted discount coupons and bundle offers to stimulate spending.",
                    "Highlight premium quality and value-for-money messaging.",
                    "Introduce a referral programme with attractive incentives.",
                ]
                channels = ["Email campaigns", "Direct mail with premium branding",
                            "Social media retargeting", "SMS offers"]

            elif low_income and high_spending:
                interpretation = (
                    f"Cluster {p['cluster_id']} has lower income (avg ${p['mean_income']}k) "
                    f"but high spending tendencies (avg score {p['mean_spending_score']}). "
                    f"They are enthusiastic shoppers and may respond well to instalment "
                    f"plans and affordable luxury positioning."
                )
                strategy = [
                    "Promote buy-now-pay-later and instalment payment options.",
                    "Offer budget-friendly product lines that feel premium.",
                    "Run flash sales and time-limited deals to match their buying behaviour.",
                ]
                channels = ["Social media ads", "Mobile App notifications",
                            "Influencer partnerships", "In-store promotions"]

            elif low_income and low_spending:
                interpretation = (
                    f"Cluster {p['cluster_id']} has lower income (avg ${p['mean_income']}k) "
                    f"and low spending (avg score {p['mean_spending_score']}). They are "
                    f"price-sensitive and less engaged. Focus on low-cost retention and "
                    f"gradual relationship building."
                )
                strategy = [
                    "Offer basic loyalty points and small rewards for repeat visits.",
                    "Run seasonal clearance and value-pack promotions.",
                    "Use low-cost digital channels to maintain brand awareness.",
                ]
                channels = ["Email newsletters", "Social media organic content",
                            "In-store flyers", "Community events"]

            else:
                # Average / mixed segments
                interpretation = (
                    f"Cluster {p['cluster_id']} represents average customers with moderate "
                    f"income (avg ${p['mean_income']}k) and moderate spending "
                    f"(avg score {p['mean_spending_score']}). They form the core customer "
                    f"base and can be nudged towards higher engagement."
                )
                strategy = [
                    "Implement a tiered loyalty programme to encourage progression.",
                    "Use personalised product recommendations based on purchase history.",
                    "Run periodic engagement surveys to understand their needs better.",
                ]
                channels = ["Email", "Mobile App", "Social media",
                            "In-store displays"]

            recommendations.append({
                "cluster_id": p["cluster_id"],
                "segment_name": segment,
                "interpretation": interpretation,
                "strategy": strategy,
                "channels": channels,
            })

        print(f"[RecommendationEngine] Generated recommendations for {len(recommendations)} segments.")
        return recommendations

    # ------------------------------------------------------------------ #
    #  Overall insights
    # ------------------------------------------------------------------ #
    def generate_overall_insights(self, profiles: list, n_clusters: int) -> list:
        """
        Derive 3-5 high-level business insights from the profiles.

        Returns a list of insight strings.
        """
        insights = []

        # 1. Number of distinct segments
        insights.append(
            f"The customer base naturally segments into {n_clusters} distinct groups, "
            f"each requiring a differentiated engagement strategy."
        )

        # 2. Largest segment
        largest = max(profiles, key=lambda p: p["count"])
        insights.append(
            f"The largest segment is '{largest.get('segment_name', 'Cluster ' + str(largest['cluster_id']))}' "
            f"with {largest['count']} customers ({largest['percentage']}% of the base)."
        )

        # 3. Highest spenders
        top_spender = max(profiles, key=lambda p: p["mean_spending_score"])
        insights.append(
            f"'{top_spender.get('segment_name', 'Cluster ' + str(top_spender['cluster_id']))}' "
            f"has the highest average spending score ({top_spender['mean_spending_score']}), "
            f"making it the prime target for upselling and cross-selling."
        )

        # 4. Gender skew observation
        total_male = sum(p["male_count"] for p in profiles)
        total_female = sum(p["female_count"] for p in profiles)
        if total_male > total_female:
            dominant, ratio = "male", round(total_male / max(total_female, 1), 2)
        else:
            dominant, ratio = "female", round(total_female / max(total_male, 1), 2)
        insights.append(
            f"Overall, the customer base skews slightly {dominant} "
            f"({total_male} male vs {total_female} female, ratio {ratio}:1). "
            f"Gender-balanced marketing should be considered."
        )

        # 5. Income spread
        min_income_seg = min(profiles, key=lambda p: p["mean_income"])
        max_income_seg = max(profiles, key=lambda p: p["mean_income"])
        income_gap = round(max_income_seg["mean_income"] - min_income_seg["mean_income"], 2)
        insights.append(
            f"There is a ${income_gap}k average-income gap between the highest-earning "
            f"segment ('{max_income_seg.get('segment_name', '')}') and the lowest "
            f"('{min_income_seg.get('segment_name', '')}'), suggesting a need for "
            f"tiered product and pricing strategies."
        )

        return insights

    # ------------------------------------------------------------------ #
    #  Targeting strategies
    # ------------------------------------------------------------------ #
    def get_targeting_strategies(self, profiles: list) -> dict:
        """
        Return a dict mapping each segment_name to a concise targeting
        strategy description.
        """
        strategies = {}
        for p in profiles:
            segment = p.get("segment_name", "Unknown")
            seg_lower = segment.lower()

            if "high income" in seg_lower and "high spenders" in seg_lower:
                strategies[segment] = (
                    "Retain and reward: deliver exclusive experiences, premium "
                    "loyalty tiers, and personalised high-touch service to maximise "
                    "lifetime value."
                )
            elif "high income" in seg_lower and "low spenders" in seg_lower:
                strategies[segment] = (
                    "Activate and convert: use aspirational messaging, limited-time "
                    "offers, and product bundling to unlock their spending potential."
                )
            elif "low income" in seg_lower and "high spenders" in seg_lower:
                strategies[segment] = (
                    "Sustain and support: offer flexible payment plans, value packs, "
                    "and budget-friendly alternatives to keep them engaged without "
                    "over-extending."
                )
            elif "low income" in seg_lower and "low spenders" in seg_lower:
                strategies[segment] = (
                    "Nurture gradually: maintain low-cost digital touchpoints, "
                    "provide entry-level incentives, and monitor for upward mobility "
                    "signals."
                )
            else:
                strategies[segment] = (
                    "Engage and grow: use personalised recommendations, mid-tier "
                    "loyalty benefits, and targeted promotions to move them into "
                    "higher-value segments."
                )

        return strategies
