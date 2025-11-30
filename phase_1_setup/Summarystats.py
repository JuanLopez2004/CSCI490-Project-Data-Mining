import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Datasets\\training_data.csv')

# Basic info
print(f"Dataset size: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
# Summary statistics
print(df.describe())
print(df.dtypes)
print(df.head())
print(df['sex'])

death_events = df[df['DEATH_EVENT'] == 1]

print(death_events)
death_events_over_62 = death_events[death_events["age"] > 62]
death_events.plot(kind="hist", y="age", bins=10, title="Age distribution of death events over 62")
plt.show()


print(death_events.describe())
# Target distribution
#df['DEATH_EVENT'].value_counts().plot(kind='bar')
#plt.title('Survival vs Death')
#plt.show().
def sumstatistics():
    # Basic info
    print(f"Dataset size: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    # Summary statistics
    print(df.describe())
    print(df.dtypes)
    print(df.head())
    print(df['sex'])