# Risk weights v2: the team's weight set

**Version:** `v2-team-proposed` (replaces `v1-placeholder` as the default; v1 is kept unchanged so old scores can be reproduced)

**Status: team-proposed and not yet validated.** Nobody with legal or procurement expertise has
reviewed these values. They are the team's reading of the CUAD category definitions, checked
against the training data only for "does this rule fire at a sensible rate". Nothing here has been
compared with a reviewer's own risk judgement. Please treat every number as a starting point to
correct, not a finding. The scores are not legal advice.

This document is written so it can be corrected line by line. If you mark up the tables below
(change a tier, strike a phrase, change a threshold), the team will turn your edits into a new
version (for example `v3-team-revised`). Versions are never edited in place, because every
score records which version produced it.

---

## 1. How a clause gets its rating

The classifier first finds clauses and labels each with one of the 41 CUAD categories. The risk
layer then rates each found clause Low, Medium or High from four signals, each between 0 and 1:

| Signal | What it asks | Weight in v2 (v1) |
|---|---|---|
| **Category severity** | How serious is this kind of clause on its own? A fixed tier per category (table in section 2), reduced when a protective clause elsewhere in the contract softens it. | 60% (55%) |
| **One-sidedness** | Does the quoted sentence give one side unchecked power ("sole discretion", "without cause"), or is it balanced ("either party", "reasonable")? | 10% (15%) |
| **Magnitude** | Is the obligation unusually broad ("perpetual", "irrevocable") or long (a 3-year non-compete)? | 10% (15%) |
| **Contract context** | Does the rest of the contract make this clause worse, through a risky combination or a missing protection? | 20% (15%) |

The weighted sum is the clause score (0 to 1). **Medium starts at 0.25 and High at 0.525.**

Severity tiers: **None** 0, **Low** 0.25, **Medium** 0.5, **High** 0.75, **Critical** 1.0.

What that arithmetic means in practice (the thresholds were chosen so these statements hold):

| Clause | Score | Rating |
|---|---|---|
| A plain None-tier clause (e.g. Parties) | 0 | Low |
| A plain Low-tier clause | 0.15 | Low |
| A Low-tier clause with one full-strength warning sign (unbalanced one-sided wording, or a duration past its limit) | 0.25 | **Medium** |
| A plain Medium-tier clause | 0.30 | Medium |
| A Medium-tier clause with both quote warning signs | 0.50 | Medium |
| A plain High-tier clause | 0.45 | Medium |
| A High-tier clause with one full-strength warning sign | 0.55 | **High** |
| A Critical clause (Uncapped Liability) | 0.60 or more | High |
| Uncapped Liability next to a liability cap, with an insurance requirement | 0.48 | Medium |

**Team judgement:** a plain High-tier clause (a non-compete, an exclusivity clause) rates Medium,
not High, and needs one aggravating feature to reach High. We did this so "High" stays rare enough
to mean "open this first". About 4% of found clauses rate High on the validation contracts.

**Why the weights moved.** One-sidedness and magnitude read a single quoted sentence chosen by a
simple evidence extractor, which is right only about half the time (about 55% of its quotes
overlap the annotated answer). Contract context only depends on which clause types were found, so
it is more reliable. We moved weight from the two quote signals to context and severity.

---

## 2. Category tiers (all 41)

v1 set a tier for only 9 categories and gave the other 32 a default of Medium. v2 sets every one.
Tiers assume a **neutral reviewer**: we do not know which side of the contract the reader is on,
so "risk" means "a reviewer should read this carefully", not "this is bad for our client".

| Category | Tier | Why |
|---|---|---|
| Document Name | None | Identifies the document; creates no obligation. |
| Parties | None | Identifies who signed; creates no obligation. |
| Agreement Date | None | A date for the record; creates no obligation. |
| Effective Date | None | Says when obligations start, not what they are. |
| Expiration Date | Low | The length of the commitment; a very long fixed term is caught by the duration rule. |
| Renewal Term | Medium | Automatic renewal can lock a party in past the term it planned for. |
| Notice Period To Terminate Renewal | Low | A calendar item to diarise; the lock-in risk is carried by Renewal Term. |
| Governing Law | Low | Matters mainly when the law or forum is unfamiliar, which the system cannot judge. |
| Most Favored Nation | High | Ties the seller's pricing to every other customer deal and invites audits and repricing claims. |
| Non-Compete | High | Restricts what business a party may do, often beyond the contract term. |
| Exclusivity | High | Shuts out other suppliers or customers, so the party depends on one counterparty. |
| No-Solicit Of Customers | Medium | Limits who a party may sell to; narrower than a non-compete. |
| Competitive Restriction Exception | Low | A carve-out that softens a restriction; worth a read to confirm it is wide enough. |
| No-Solicit Of Employees | Low | Common, usually mutual and time-limited; a long period is caught by the duration rule. |
| Non-Disparagement | Low | Restricts statements about the counterparty; rarely carries money exposure. |
| Termination For Convenience | Medium | The counterparty can walk away without cause, so revenue and investment are uncertain. |
| Rofr/Rofo/Rofn | Medium | Gives the counterparty first claim on future deals, which can chill or delay other buyers. |
| Change Of Control | Medium | A merger or sale of the business can need consent or end the contract. |
| Anti-Assignment | Low | A consent requirement found in most contracts; mainly matters in a transaction. |
| Revenue/Profit Sharing | Medium | An ongoing payment obligation that is hard to verify and often audited. |
| Price Restrictions | Medium | Limits the ability to raise or lower prices as costs and markets move. |
| Minimum Commitment | High | Money is owed whether or not the goods or services are needed (take-or-pay). |
| Volume Restriction | Medium | Exceeding a usage threshold triggers fees or needs consent. |
| IP Ownership Assignment | High | A party permanently gives up ownership of intellectual property it creates. |
| Joint IP Ownership | Medium | Shared ownership makes licensing and enforcement depend on the co-owner. |
| License Grant | Low | A core commercial term present in most licensing deals; a perpetual or irrevocable grant is caught by the broad-scope words. |
| Non-Transferable License | Low | Limits flexibility in a transaction but is standard licensing practice. |
| Affiliate License-Licensor | Low | Widens which licensor-group IP is covered; mainly a scope question. |
| Affiliate License-Licensee | Low | Widens who may use the licence; mainly a scope question. |
| Unlimited/All-You-Can-Eat License | Medium | Usage is uncapped, so the licensor cannot charge for growth. |
| Irrevocable Or Perpetual License | Medium | The licence cannot be taken back, even if the relationship ends badly. |
| Source Code Escrow | Low | Mostly protective for the licensee; the licensor's risk is a broad release trigger. |
| Post-Termination Services | Medium | Obligations survive the end of the contract, including transition, payments or last-buy. |
| Audit Rights | Low | An operational burden that is standard in commercial contracts. |
| Uncapped Liability | Critical | Exposure has no ceiling, so one breach can cost more than the contract is worth. |
| Cap On Liability | Low | Limits exposure for the protected party; reduces recovery for the other side but is market standard. |
| Liquidated Damages | High | A fixed payment for breach or early termination is owed without proving loss. |
| Warranty Duration | Low | The length of warranty cover; a short window is caught by the duration rule. |
| Insurance | Low | A compliance cost; it mainly protects the counterparty. |
| Covenant Not To Sue | Medium | A party gives up the right to challenge IP or bring certain claims. |
| Third Party Beneficiary | Medium | Someone who did not sign the contract can enforce it. |

Tier counts: None 4, Low 16, Medium 14, High 6, Critical 1.

---

## 3. Contract context rules

A rule looks at which other clause types were found in the same contract. A **raising** rule adds
to the context signal; a **softening** rule reduces the clause's severity by the stated share. The
"how often" column is the share of training contracts containing the target clause where the rule
applies (326 training contracts, from CUAD's annotations).

| Clause | Condition | Effect | Why | How often |
|---|---|---|---|---|
| Uncapped Liability | no Insurance clause | raise 0.5 | Unlimited exposure with nothing behind the promise to pay. | 43 of 84 |
| Uncapped Liability | a Cap On Liability clause exists | soften 20% | **New.** With a cap elsewhere, "uncapped" is usually a carve-out for specific breaches (IP, confidentiality), not unlimited exposure overall. Also hedges a known classifier error (below). | 84 of 84 |
| Liquidated Damages | a Cap On Liability clause exists | soften 20% | A cap usually bounds the total of fixed damages. | 31 of 41 |
| Liquidated Damages | no Cap On Liability | raise 0.5 | Fixed damages with no overall cap. | 10 of 41 |
| Non-Compete | a Competitive Restriction Exception exists | soften 20% | A carve-out narrows the restriction. | 27 of 75 |
| Exclusivity | a Competitive Restriction Exception exists | soften 20% | **New.** CUAD defines the exception for exclusivity as well. | 35 of 120 |
| No-Solicit Of Customers | a Competitive Restriction Exception exists | soften 20% | **New.** Same reason. | 12 of 22 |
| Exclusivity | a Minimum Commitment exists | raise 0.5 | Exclusive and obliged to buy a minimum: cannot go elsewhere or buy less. | 58 of 120 |
| Minimum Commitment | an Exclusivity clause exists | raise 0.5 | The same lock-in, seen from the purchase obligation. | 58 of 108 |
| Minimum Commitment | no Termination For Convenience | raise 0.3 | No early exit, so the minimum is owed for the whole term. | 71 of 108 |
| Renewal Term | no Termination For Convenience | raise 0.3 | **New.** Auto-renewal with no exit means a missed notice window locks in another full term. | 70 of 120 |
| Anti-Assignment | a Change Of Control clause exists | raise 0.3 | Together they mean a merger or sale needs the counterparty's consent. | 77 of 249 |

**The Uncapped vs Cap On Liability problem.** Our error analysis found that the classifier often
labels ordinary cap language as "Uncapped Liability" (83% of the reranker's false Uncapped
detections were on Cap On Liability text). That flips the risk direction: a protective clause is
reported as the most severe one. On the validation contracts, 19 contracts were flagged for
Uncapped Liability but only 11 really have one; 9 of the 22 flagged passages were cap language.
What v2 does about it:

- The new softening rule lowers Uncapped Liability from 0.60 to 0.48 whenever a cap is also found.
  In CUAD every contract with an Uncapped Liability clause also has a cap (84 of 84 training
  contracts), so in practice this rule always applies and Uncapped Liability behaves like a
  "High-plus" tier. It keeps the full Critical score for a contract where no cap is found at all.
  A real carve-out without an insurance requirement still rates High (0.58).
- The Liquidated Damages rules depend on detecting a cap. Cap On Liability is a common, better-detected
  category, so we kept them, but a missed cap would wrongly raise Liquidated Damages.
- What v2 cannot do: scoring cannot tell a real uncapped clause from mislabelled cap language. The
  UI should show the quote for every High-rated Uncapped Liability clause so a reviewer can check it.

---

## 4. Phrase lists (one-sidedness and broad scope)

Phrases match at the start of a word and ignore case, so "mutual" also matches "mutually". Rates
below are from CUAD's annotated answers in the 326 training contracts (9,175 answers; 6,620
excluding names and dates).

**One-sidedness** = (warning phrases found) / (warning + balancing phrases found). It is 0 when no
warning phrase is found, so balancing words alone never add risk.

**Warning phrases (one-sided wording).** At least one appears in 5.9% of substantive answers (v1:
5.7%). They concentrate where expected: 65% of Termination For Convenience answers, 26% of
Covenant Not To Sue, 15% of No-Solicit Of Customers, under 3% of License Grant.

| Phrase | Answers (of 9,175) | Note |
|---|---:|---|
| at any time | 175 | |
| for any reason | 85 | |
| sole discretion | 67 | |
| without cause | 40 | also matches "with or without cause" |
| shall not be liable | 24 | mostly in Cap On Liability; mutual versions are balanced by "neither party" |
| waives | 12 | |
| absolute discretion | 5 | also matches "sole and absolute discretion" |
| without notice | 5 | |
| without liability | 5 | |
| reserves the right | 9 | **new** |
| forfeit | 5 | **new** |
| unilateral | 4 | **new** (also "unilaterally") |
| in its discretion | 3 | **new** |
| irrevocably waive | 3 | **new** |
| for any or no reason | 1 | **new**, rare but unambiguous |

Rare phrases are kept because they cost nothing when absent and are unambiguous when present.

**Balancing phrases.** At least one appears in 27.3% of substantive answers (v1: 31.9%).
Kept: either party, each party, mutual, reasonable, provided that, to the extent, prior written
notice, good faith. Changes:

- **Added "neither party"** (188 answers, mostly Anti-Assignment and Cap On Liability): "neither
  party shall be liable" is mutual, and v1 scored it as fully one-sided.
- **"unreasonably withheld" widened to "unreasonably"** (82 to 111 answers), to catch "not to be
  unreasonably conditioned or delayed".
- **Added "cure"** (29 answers): an opportunity to fix a breach before termination.
- **Removed "subject to"** (662 answers, the most common phrase in either list). Over a third of
  its uses are the boilerplate "subject to the terms and conditions of this Agreement", mostly in
  License Grant. It is a cross-reference, not a sign of balance.

**Broad-scope words (magnitude = 1 when present).** Kept: perpetual, perpetuity, irrevocable,
unlimited. Present in 4.8% of substantive answers (v1: 6.7%).

- **Removed "worldwide" and "throughout the world"** (263 and 44 answers). They appear in 22% of
  License Grant answers and 35% to 44% of affiliate licence answers, so with v2's lower Medium
  threshold every worldwide licence would have rated Medium. Territory is a commercial term
  more than a risk. The cost: a worldwide non-compete is no longer marked as broad (only 2 of 159
  non-compete answers say worldwide). See question 4.

---

## 5. Duration rules (magnitude)

Each rule gives a "fine" duration (scores 0) and a "too long" duration (scores 1), with a straight
line between. Two rules are reversed: a short period is the risk. The rates are from training
answers in that category.

| Category | Scores 0 at | Scores 1 at | Why | Training evidence |
|---|---|---|---|---|
| Non-Compete | 6 months | 3 years | Up to 6 months is common; 3 years or more risks being unenforceable and is a heavy restriction. | 23% of answers state a period, median 1 year; 15% score above 0 |
| No-Solicit Of Employees | 1 year | 3 years | 12 months is typical. | 49% state a period, median 1 year; 12% above 0 |
| No-Solicit Of Customers | 1 year | 3 years | 12 months is typical. | 26% state a period, median 1 year; 6% above 0 |
| Exclusivity | 1 year | 5 years | A five-year exclusive ties the party to one counterparty for a business cycle. | only 6% state a period; 2% above 0 |
| Renewal Term | 1 year | 5 years | Annual renewal is the norm. | 83% state a period, median 1 year; 30% above 0 |
| Post-Termination Services | 90 days | 2 years | A 90-day transition is routine. | 27% state a period, median 6 months; 17% above 0 |
| Warranty Duration (short is risky) | 1 year | 90 days | A short warranty or inspection window leaves little time to find defects. Many CUAD answers here are inspection/rejection windows, which carry the same risk. | 36% state a period, median 90 days; 25% above 0 |
| Expiration Date | 5 years | 10 years | **New.** Initial terms up to 5 years are common (three quarters of stated terms are 5 years or less). | 49% state a period, median 2 years; 7% above 0 |
| Termination For Convenience (short is risky) | 6 months | 30 days | **New.** 30 days' notice to walk away leaves little time to replace the business. | 69% state a period, median 75 days; 53% above 0 |

**Removed: Notice Period To Terminate Renewal.** The duration reader scores every period in the
quoted text and keeps the worst. Notice-period answers almost always restate the renewal term in
the same sentence ("renews for successive one-year terms unless either party gives 90 days'
notice"), and 45 of 77 such answers contain a period of a year or more. v1's rule therefore gave
73% of these clauses the maximum score, mostly for the length of the renewal, not the notice. A
fix needs a change to how durations are read (pick the period nearest the word "notice"), which
would be a versioned code change. Until then the category relies on its tier.

The duration reader does not read money amounts, so cap sizes and fee levels are not scored.

---

## 6. Contract ranking

Unchanged from v1: a contract's triage score is half its single worst clause plus half the average
of its three worst clauses (counting each clause type once). The worst clause makes sure one severe
clause always surfaces the contract; the average lifts contracts with several risky clauses.

---

## 7. What changed from v1

1. All 41 categories have an explicit tier and a one-line reason (v1: 9, the rest defaulted to Medium).
2. Weights: severity 55% to 60%, context 15% to 20%, one-sidedness and magnitude 15% to 10% each.
3. Thresholds: Medium 0.35 to 0.25, High 0.55 to 0.525, so each tier's plain rating matches its name
   (except High tier, which rates Medium until something aggravates it).
4. Four new context rules (Uncapped Liability softened by a cap; exclusivity and customer
   no-solicit softened by an exception; renewal without a convenience exit raised).
5. Phrase lists: six warning phrases added; "neither party", "unreasonably" and "cure" added and
   "subject to" removed from the balancing list; "worldwide" and "throughout the world" removed
   from broad scope.
6. Duration rules: Notice Period removed; Expiration Date and Termination For Convenience added.
7. Every tier and rule now carries its reason inside the configuration file itself.

---

## 8. Effect on the 82 validation contracts

Measured with the current TF-IDF classifier (thresholds fit on separate calibration contracts).
The test set was not used.

**Ratings of the 1,529 detected clauses**

| | High | Medium | Low |
|---|---:|---:|---:|
| v1 | 40 (2.6%) | 225 (14.7%) | 1,264 (82.7%) |
| v2 | 62 (4.1%) | 581 (38.0%) | 886 (57.9%) |

With perfect detection (CUAD's annotated answers, 2,005 clauses) v2 gives 3.7% High, 30.4%
Medium, 65.8% Low (v1: 1.4%, 16.5%, 82.1%). Under v1, a plain Medium-tier clause rated Low, which
is why so much fell into Low. By contract (rated by its worst clause): v1 25 High / 29 Medium /
28 Low; v2 29 High / 39 Medium / 14 Low.

**Agreement with the perfect-detection reference.** This compares the ranking from the classifier's
detections with the ranking the same weights would give if every clause were detected perfectly.
It measures how much detection errors disturb the ranking. **It is not a measure of whether the
ranking is right**; only a reviewer's ranking can tell us that.

| | Rank correlation | Shared top-10 | Largest rank move |
|---|---:|---:|---:|
| v1 | 0.81 | 5 | 55 |
| v2 | 0.78 | 2 | 50 |

v2 agrees slightly less. Most of the drop comes from the new context rules, each of which depends on
detecting a second clause type correctly; with v1's rules in their place the correlation would be
about 0.81. The top of the list is tightly packed (the top 12 contracts score within about 0.1 of
each other), so the top-10 overlap moves a lot for small changes. We kept the rules because they follow from the category definitions, not from this
number, and we did not tune any value to raise it.

**Stability.** Moving any one signal weight by 10 points changes the v2 ranking very little (rank
correlation with the base ranking 0.975 to 0.998; v1 0.968 to 0.998). Moving a rating threshold
by 0.05 changes 7 to 42 of the 1,529 clause ratings (v1: 1 to 90), so v2's ratings are less
sensitive to the exact Medium threshold.

**How the values were chosen.** Phrase lists and duration rules were checked on the training
contracts only. The validation contracts were used to check that the rating spread is not
degenerate. Two choices were made after looking at that spread: the Medium threshold moved from
0.275 to 0.25 (at 0.275 no warning sign could ever lift a Low-tier clause to Medium, so the
duration rules for Low-tier categories did nothing), and "worldwide" was removed (see section 4).

---

## 9. Known limitations

- **Not validated against human judgement.** Nothing here has been compared with a reviewer's own ranking.
- **Neutral perspective.** The tiers cannot tell whether the reader is the licensor or licensee,
  buyer or seller. Several categories (Cap On Liability, Source Code Escrow, Insurance,
  Unlimited License) are protective for one side and a risk for the other.
- **Evidence quotes are weak.** One-sidedness and magnitude read one sentence picked by a simple
  extractor; about a quarter of detected clauses have no verified quote and score on severity and
  context only.
- **Classifier errors carry through,** especially Uncapped vs Cap On Liability (section 3).
- **Durations:** the worst period in a quote is used, money amounts are ignored, and phrases apply
  to every category (a word cannot be "broad" for one category only).

---

## 10. Calls that are team judgement (please check these first)

1. Six High-tier categories: Most Favored Nation, Non-Compete, Exclusivity, Minimum Commitment,
   IP Ownership Assignment, Liquidated Damages.
2. Covenant Not To Sue and Third Party Beneficiary at Medium (either could be argued Low or High).
3. Cap On Liability at Low, treating it as protective.
4. A plain High-tier clause rating Medium (section 1).
5. The softening of Uncapped Liability when a cap exists.
6. The 20% size of every softening rule and the 0.3 / 0.5 size of every raising rule.
7. The "fine" and "too long" cut-offs in section 5.
8. Dropping "worldwide" from the broad-scope words.

## 11. Open questions for the team

1. **Tiers:** which rows in the section 2 table should move, and to what?
2. **Perspective:** should we score from one side (for example, always the customer or always
   Accenture's client)? That would change several tiers and is the biggest open design question.
3. **Uncapped liability:** is an uncapped carve-out for IP infringement or confidentiality, inside a
   contract that otherwise caps liability, a High or a Critical concern?
4. **Worldwide:** should territory count as broad scope for non-competes and exclusivity only?
   That needs category-specific phrase lists, a small code change.
5. **Durations:** are the cut-offs in section 5 in line with practice? Is a
   30-day convenience termination notice really the riskiest case?
6. **Missing rules:** which combinations of clauses matter first when triaging a
   contract that the context rules don't cover?
7. **High rate:** about 4% of clauses and 35% of contracts get at least one High. Is that the
   right proportion for "open this first"?
8. **Validation:** the weights have no human validation. The team decided not to collect hand
   rankings, so agreement with a real reviewer is unmeasured and is reported as a limitation.
