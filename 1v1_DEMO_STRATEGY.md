# フレイムウォール 1v1 Demo Strategy Guide 💥

## The Nexto Weakness Exploit

Nexto has a fatal flaw in 1v1: **It sits in goal waiting for you to shoot**, then makes the save. This defensive strategy is predictable and exploitable!

## The フレイムウォール Counter-Strategy

### Phase 1: Slow Dribble 🐌
- When opponent is camping in goal (within 1500 units of goal line)
- **Beta drops to 0.8** - careful, controlled dribbling
- Slowly advance the ball toward their goal
- Maintain close ball control (< 300 units)
- Ball must be mid-field or closer (> 3000 units from opponent goal)

### Phase 2: Demo Setup 💣
- Continue dribbling slowly to draw closer
- Opponent stays in goal (Nexto isn't trained to challenge)
- Once within **2000 units** of opponent...
- **TRIGGER DEMO MODE!**

### Phase 3: BOOM! 💥
- **Beta spikes to 2.0** - ULTRA AGGRESSIVE
- Bot abandons ball temporarily
- Goes FULL SPEED at opponent
- Nexto has no demo-dodge training
- Nexto gets demolished out of goal

### Phase 4: Free Goal 🎯
- With opponent demolished, goal is EMPTY
- Simply push ball into net
- Easy score!

## Technical Implementation

```python
# Detection Phase
if opponent_distance_to_goal < 1500:  # Camping detected
    if my_distance_to_ball < 300:      # We have ball control
        if ball_distance_to_opp_goal > 3000:  # Safe distance
            slow_dribble_mode = True
            beta = 0.8  # Careful dribbling

# Demo Phase
if distance_to_opponent < 2000:        # In striking range
    if opponent_in_goal_ticks > 60:    # They've been camping
        beta = 2.0  # MAXIMUM AGGRESSION - GO FOR DEMO!
```

## Settings Optimized for This Strategy

| Setting | Value | Purpose |
|---------|-------|---------|
| **beta (base)** | 1.5 | High baseline aggression |
| **beta (dribble)** | 0.8 | Careful ball control |
| **beta (demo)** | 2.0 | ULTRA AGGRESSIVE demo mode |
| **tick_skip** | 4 | Super fast reactions (30 actions/sec) |
| **kickoffs** | Consistent | Reliable 1v1 starts |

## Why This Works Against Nexto

1. **Nexto waits in goal** - predictable defensive position
2. **Nexto doesn't challenge** - won't interrupt your dribble
3. **Nexto can't dodge demos** - not trained for it
4. **Nexto is stationary** - easy target for demolition
5. **Post-demo goal is empty** - free score

## Additional 1v1 Optimizations

### Goalie Mode
- When defending, bot still uses aggressive positioning
- Beta=1.5 baseline keeps pressure even on defense
- Fast tick_skip=4 means quick saves and clears

### Boost Feathering
- Dynamic beta adjustments help with boost management
- Beta=0.8 during dribbles = smoother boost usage
- Beta=2.0 for demos = full boost commitment

### Recovery
- If demo attempt fails, beta returns to 1.5
- Quick reaction time (tick_skip=4) allows fast recovery
- Re-enter slow dribble mode if opponent retreats to goal

## Pro Tips

1. **Be Patient** - Let opponent stay in goal, don't rush
2. **Maintain Control** - Keep ball close during dribble phase
3. **Commit to Demo** - When beta hits 2.0, full send!
4. **Follow Through** - After demo, immediately score
5. **Repeat** - This works EVERY time opponent camps goal

## Watch It Live!

See フレイムウォール demolish Nexto on stream:
- **Twitch:** https://www.twitch.tv/aauuzy
- **YouTube:** https://www.youtube.com/@aauuzy

## Comparison to Standard Play

| Situation | Standard Bot | フレイムウォール |
|-----------|--------------|----------------|
| Opponent in goal | Shoot on goal | Slow dribble → Demo |
| On defense | Clear ball | Aggressive challenge |
| 50/50 situation | Calculated | COMMIT (beta=1.5) |
| Demo opportunity | Occasional | **ALWAYS** (when opponent camps) |

---

**TL;DR:** Nexto sits in goal → We slow dribble → We BOOM Nexto → Free goal → Repeat → Win 🔥💥
