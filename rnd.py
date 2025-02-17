import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Let's assume the DataFrame is already loaded in the variable df
# We'll create a copy to work with
df = pd.read_csv('FacilityCare-Walkway.csv')
data = df.copy()

# 1. Bar Chart of Top 10 Sites with Most Jobs
top_sites = data['SITE ADDRESS'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_sites.values, y=top_sites.index, palette='viridis')
plt.title('Top 10 Sites with Most Jobs')
plt.xlabel('Number of Jobs')
plt.ylabel('Site Address')
plt.tight_layout()
plt.show()

# 2. Histogram of Material Usage Distribution
plt.figure(figsize=(10,6))
sns.histplot(data=data, x='MATERIAL USAGE', kde=True, bins=20, color='skyblue')
plt.title('Distribution of Material Usage')
plt.xlabel('Material Usage')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Scatter Plot: Relationship between # CREW MEMBERS and MATERIAL USAGE
scatter_data = data.dropna(subset=['MATERIAL USAGE'])
plt.figure(figsize=(10,6))
sns.scatterplot(data=scatter_data, x='# CREW MEMBERS', y='MATERIAL USAGE', hue='# CREW MEMBERS', palette='deep', s=100)
plt.title('Relationship between # CREW MEMBERS and Material Usage')
plt.xlabel('Number of Crew Members')
plt.ylabel('Material Usage')
plt.tight_layout()
plt.show()

# 4. Jobs Over Time
# Convert the START DATE to a datetime object; the format in the data appears as '4-Dec-24', etc.
data['START DATE'] = pd.to_datetime(data['START DATE'], format='%d-%b-%y', errors='coerce')

# Group by start date and count the jobs
jobs_over_time = data.groupby('START DATE').size().reset_index(name='job_counts')
plt.figure(figsize=(10,6))
sns.lineplot(data=jobs_over_time, x='START DATE', y='job_counts', marker='o')
plt.title('Jobs Over Time')
plt.xlabel('Start Date')
plt.ylabel('Number of Jobs')
plt.tight_layout()
plt.show()

# 5. Top 10 Crew Members by Material Usage
top_crew_members = data.groupby(['# CREW MEMBERS', 'MATERIAL USAGE']).size().reset_index(name='job_counts')
plt.figure(figsize=(10,6))
sns.barplot(data=top_crew_members, x='job_counts', y='# CREW MEMBERS', palette='viridis')
plt.title('Top 10 Crew Members by Material Usage')
plt.xlabel('Number of Jobs')
plt.ylabel('Crew Member')
plt.tight_layout()
plt.show()

# 6. Top 10 Sites by Total Material Usage
top_sites = data.groupby('SITE ADDRESS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_sites.values, y=top_sites.index, palette='viridis')
plt.title('Top 10 Sites by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Site Address')
plt.tight_layout()
plt.show()

# 7. Top 10 Crew Members by Total Material Usage
top_crew_members = data.groupby('# CREW MEMBERS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_crew_members.values, y=top_crew_members.index, palette='viridis')
plt.title('Top 10 Crew Members by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Crew Member')
plt.tight_layout()
plt.show()

# 8. Top 10 Sites by Total Material Usage
top_sites = data.groupby('SITE ADDRESS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_sites.values, y=top_sites.index, palette='viridis')
plt.title('Top 10 Sites by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Site Address')
plt.tight_layout()
plt.show()

# 9. Top 10 Crew Members by Total Material Usage
top_crew_members = data.groupby('# CREW MEMBERS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_crew_members.values, y=top_crew_members.index, palette='viridis')
plt.title('Top 10 Crew Members by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Crew Member')
plt.tight_layout()
plt.show()

# 10. Top 10 Crew Members by Total Material Usage
top_crew_members = data.groupby('# CREW MEMBERS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_crew_members.values, y=top_crew_members.index, palette='viridis')
plt.title('Top 10 Crew Members by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Crew Member')
plt.tight_layout()
plt.show()

# 11. Top 10 Crew Members by Total Material Usage
top_crew_members = data.groupby('# CREW MEMBERS')['MATERIAL USAGE'].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_crew_members.values, y=top_crew_members.index, palette='viridis')
plt.title('Top 10 Crew Members by Total Material Usage')
plt.xlabel('Total Material Usage')
plt.ylabel('Crew Member')
plt.tight_layout()
plt.show()

