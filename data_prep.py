import pandas as pd;
athletes = pd.read_csv("athlete_events.csv")
athletes = athletes[athletes["Season"] == "Summer"]

def team_summary(data):
    return pd.Series({
        'team': data.iloc[0,:]["NOC"],
        'country': data.iloc[-1,:]["Team"],
        'year': data.iloc[0,:]["Year"],
        'events': len(data['Event'].unique()),
        'athletes': data.shape[0],
        'age': data["Age"].mean(),
        'height': data['Height'].mean(),
        'weight': data['Weight'].mean(),
        'medals': sum(~pd.isnull(data["Medal"]))
    })

team = athletes.groupby(["NOC", "Year"]).apply(team_summary)

team = team.reset_index(drop=True)
team = team.dropna()

def prev_medals(data):
    data = data.sort_values("year", ascending=True)
    data["prev_medals"] = data["medals"].shift(1)
    data["prev_3_medals"] = data.rolling(3, closed="left", min_periods=1).mean()["medals"]
    return data

team = team.groupby(["team"]).apply(prev_medals)
team = team.reset_index(drop=True)
team = team[team["year"] > 1960]
team = team.round(1)

team.to_csv("teams.csv", index=False)

