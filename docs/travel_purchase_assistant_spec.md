# Product concept: Personal flight-buying assistant

## One-sentence product

A trip-aware assistant that detects a possible journey from a user's calendar (after explicit confirmation), continuously gathers comparable fare quotes, and recommends **which acceptable itinerary/cabin to book and when**, with a transparent explanation and a booking link.

The initial scenario is a London-to-San Francisco journey roughly 6–12 months ahead. That is a research target, not a promise that every fare is already published or that a forecast is reliable that far ahead.

## Product principle

Optimize for **the best acceptable trip with the least user effort**, not just the lowest headline price. The assistant should act quietly, keep track of changing options, and interrupt only when a decision needs attention.

It should never promise the absolute cheapest ticket or claim it outperforms every existing service until that has been tested against a predefined benchmark.

## What makes the product distinct

Price tracking and buy/wait predictions already exist:
- Google Flights tracks route/date and specific-flight prices, and offers alerts about notable changes and some likely price rises: https://support.google.com/travel/answer/6235879
- KAYAK describes forecasts for price movement over roughly the next seven days when adequate data exists: https://www.kayak.com/c/help/pricing/how-does-kayak-price-forecast-work.62022.faq
- Hopper explains that it uses historical and live fare observations for buy/wait recommendations, and may stay quiet when a trip is far away: https://help.hopper.com/en_us/about-our-price-predictions-Hy7cLt_Fv

A plausible differentiation is **personalized, end-to-end decision support**: calendar-aware deadlines, multiple cabins and fare families, total trip cost, explicit comfort constraints, a route-specific price history, a consistent buy/wait policy, and one calm recommendation rather than a page of alerts. Whether it is better must be demonstrated through walk-forward backtests and a transparent comparison with these baselines.

## User journey

1. **Trip candidate:** read candidate calendar events only after the user authorizes calendar access.
2. **Confirm once:** ask the user to confirm origin, destination, date or date range, trip purpose if relevant, party size, return date, flexibility, budget, cabin preferences, stops, baggage, and unacceptable itineraries.
3. **Wait for fares to appear:** when the route/date is not on sale, show “not on sale / not enough data yet,” track schedule availability if the provider supports it, and do not fabricate a price forecast.
4. **Collect comparable offers:** periodically query approved sources and save time-stamped quotes, availability, fare family, cabin, baggage, refund/change rules when available, itinerary, total price, currency, provider, and deep link.
5. **Estimate and decide:** rank current offers against user preferences, estimate price uncertainty, and calculate the expected cost of buying now versus waiting.
6. **Notify sparingly:** issue a recommendation when the action materially changes, a strong target fare appears, the availability risk becomes high, or the user's decision deadline approaches.
7. **Booking confirmation:** the user books through a verified airline/provider link; the app asks for confirmation or imports the booking if authorized.
8. **Travel mode:** stop fare alerts and present the itinerary, check-in timing, departure reminders, and important documents.

Google Calendar's `calendar.events.readonly` scope allows reading events; request the narrowest scope needed and explain the reason for access. Official scope guide: https://developers.google.com/workspace/calendar/api/auth

## The decision model

### A quote is an itinerary, not just a number

For each option j, keep at least:
- flight legs, airports, local times, total duration, layovers, and stop count;
- marketing/operating airline and cabin;
- fare family / booking conditions where available;
- fare, taxes, baggage, seat fees, payment/booking fees, and currency;
- change/cancellation rights, refundability, and connection risk where measurable;
- source, query time, quote/availability identifiers, and purchase deep link.

Do not treat “economy” as one product: basic/light, standard, flex/refundable economy, and different baggage/seat entitlements can change its real value. Premium economy, business, and first class should be compared according to explicit user preferences—not simply sorted together by sticker price.

A user preference profile should set hard constraints (e.g. maximum number of stops or maximum budget) separately from soft preferences (e.g. nonstop, preferred airline, arrive by a given time, lie-flat seat, baggage, flexibility). Show the Pareto trade-off when no single itinerary is best in every dimension.

### Dynamic programming formulation

Let t be time remaining until the user's latest acceptable booking point, and H_t the information available at that time: quotes seen, recent fare paths, days to departure, availability, date flexibility, remaining feasible itineraries, and user constraints.

For itinerary j available now, let G_j(H_t) be its generalized cost: price plus explicitly modeled add-on costs/risks and a documented penalty for violating soft preferences. Hard constraints exclude unacceptable options entirely. A basic buy/wait recursion is:

```text
V_t(H_t) = min(
    min_{j in Available(H_t)} G_j(H_t),
    c_monitor + E[V_{t-1}(H_{t-1}) | H_t]
)
```

- **Buy:** select the available itinerary with the lowest generalized cost.
- **Wait:** collect another observation, then decide with updated information.
- **Terminal condition:** at the booking deadline, select the best acceptable available offer; if none exists, report the infeasibility explicitly.
- **Booked state:** absorbing. Once the user confirms the purchase, price tracking stops unless the user explicitly enables a separate post-booking price-drop or rebooking feature.

This simplified expression is a design sketch. A production model needs a proper state transition model, action-dependent observation process, no-availability/price-change risk, and a coherent objective over price, trip quality, delay, and failure to travel.

### Price forecast and uncertainty

Start with statistical baselines before complex machine learning:
1. empirical fare distributions by route, departure date type, lead-time band, weekday/season, cabin, and itinerary type;
2. persistence and recent-trend baselines;
3. calibrated probability that a comparable acceptable fare falls by at least a threshold within a specified horizon;
4. a model of fare increases, disappearing inventory, schedule changes, and quote availability if enough data supports it.

Use dynamic programming to convert predictive distributions into actions, not as a substitute for the missing distribution. Forecast uncertainty must be visible. If quote history is sparse, say “insufficient data” and fall back to an explainable rule or an existing price alert. Do not present a fabricated point prediction with false precision.

The model must distinguish a future fare decline from a future **buying opportunity**: a cheaper fare is not useful if the itinerary is no longer available or no longer satisfies the user's constraints.

## Why 6–12 months requires special care

Airlines do not all publish schedules on the same timetable. As examples, American Airlines says its bookings are available up to 331 days in advance, while Singapore Airlines says up to 355 days:  
- https://www.aa.com/i18n/customer-service/faqs/reservations-tickets-faqs.jsp
- https://www.singaporeair.com/en_UK/bd/faq/bookings/booking-online/

The product should therefore separate:
- **pre-sale monitoring:** wait until relevant schedules and fares are available; record no-sale or missing coverage as such;
- **active price monitoring:** collect comparable quotes as soon as options are searchable;
- **deadline-aware decisioning:** adjust the cost of waiting as the user's booking deadline gets closer.

Six-to-twelve-month predictions require data across that lead-time horizon. A model trained only on the final 30–90 days cannot honestly claim reliable forecasts at 9–12 months.

## Data-access strategy and risks

Keep provider integrations behind a common `FlightSearchProvider` interface. The provider adapter must expose source coverage and limitations rather than implying completeness.

- Skyscanner's Live Prices API returns current prices from its airline and inventory partners, but API access and commercial terms must be secured before depending on it: https://developers.skyscanner.net/docs/flights-live-prices/quick-start
- Amadeus Self-Service documents important coverage limitations, including missing American Airlines, Delta, British Airways, and low-cost carriers in its self-service dataset. That is a material gap for many London–San Francisco comparisons: https://admin.developers.amadeus.com/self-service/apis-docs/guides/developer-guides/faq/

Do not build a production promise on scraping consumer websites, brittle browser automation, or a source that forbids the intended use. Do not display an option as “best across all airlines” when the data feed is partial. Log data source, query time, observed coverage, and failures.

## Evaluation plan

Use time-ordered historical replay / walk-forward backtesting. At every historical quote timestamp, the policy may see only information known at that time—never future fares. Tune on earlier periods and evaluate on later unseen periods. Split by journey/departure period to reduce leakage from near-duplicate quotes.

Compare with:
- buy the first acceptable quote;
- buy below a fixed route/cabin threshold;
- the selected price alert's recommendation where obtainable;
- hindsight minimum among observed, acceptable offers (a lower bound / oracle benchmark, not an attainable baseline).

Measure:
- actual policy cost and savings against fixed baselines;
- regret relative to the lowest acceptable price observed in the replay;
- proportion of trips for which the recommendation missed a cheaper later offer;
- last-minute infeasibility / no acceptable offer rate;
- fare availability and provider coverage;
- cabin/itinerary utility;
- forecast calibration and Brier/log score for directional or threshold probabilities;
- notification count and time spent by the user.

For every result, report route/date coverage, observation cadence, sample size, uncertainty intervals, and the fare source. A small private benchmark is not evidence that the product beats all market services. A credible public claim needs a fair, repeatable evaluation across many routes, dates, cabins, and lead-time windows.

## Build plan

### Phase 0 — Offline algorithm explorer
Build a synthetic quote simulator and an interactive buy/wait DP explorer. Test deadlines, fare drops, fare spikes, disappearing options, different risk preferences, and missing observations. This can be built before any commercial API credentials arrive.

### Phase 1 — Real quote logger
Create one route watch for LHR/LGW to SFO, date range, and user-selected alternatives. Use an approved provider, store timestamped offers, and plot historical fares by comparable itinerary/cabin/fare family. At this phase, report trends rather than promising forecasts.

### Phase 2 — Backtested recommendation engine
Build baseline forecast models and a buy/wait policy. Add no-data fallbacks and evaluate via walk-forward replay. Surface “buy / wait / insufficient evidence” with estimated cost, risk, and a short explanation.

### Phase 3 — Calendar and notifications
Integrate read-only Calendar OAuth, event confirmation, quiet hours, digest/sparsity controls, and alerts tied to the trip deadline. Calendar integration should not block development of the core decision engine.

### Phase 4 — Broader market comparison
Only after coverage, data quality, and evaluation results are credible should we expand route/cabin coverage or claim advantages over established products.

## Explicit non-goals for the first version

- No automatic ticket purchases.
- No promise of the global cheapest fare.
- No claim to beat Google Flights, KAYAK, Hopper, or every competitor before benchmark evidence exists.
- No assumption that one price API sees every airline, cabin, private fare, package, or ancillary.
- No predictions from an unprovided or unlicensed dataset.
