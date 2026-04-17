import os
import matplotlib.pyplot as plt

def save_csv(report,file_path):
    report.to_csv(file_path,index=False)

def save_excel(report,file_path):
    report.to_excel(file_path,index=False)

def make_summary(report):
    if report.empty:
        text ='## Summary\nПростой в развитии детей не найден'
        return text
    total = len(report)
    high = len(report[report['risk_level'] == 'high'])
    medium =len(report[report['risk_level'] == 'medium'])
    low = len(report[report['risk_level'] == 'low'])

    text = "# Summary\n\n"
    text += f"Всего случаев застоя: {total}\n\n"
    text += f"- High risk: {high}\n"
    text += f"- Medium risk: {medium}\n"
    text += f"- Low risk: {low}\n"

    return text
def save_summary(text,file_path):
    with open(file_path,'w',encoding = 'utf-8') as f:
        f.write(text)

def make_plot(df,child_id,domain,output_folder):
    os.makedirs(output_folder,exist_ok=True)

    part = df[(df['child_id'] == child_id) & (df['domain'] == domain)].copy()

    if part.empty:
        return
    part = part.sort_values('session_date')

    plt.figure(figsize=(8, 4))
    plt.plot(part["session_date"], part["assessment_score"], marker="o")
    plt.title(f"{child_id} - {domain}")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_domain = str(domain).replace("/", "_").replace(" ", "_")
    file_name = f"{child_id}_{safe_domain}.png"
    file_path = os.path.join(output_folder, file_name)

    plt.savefig(file_path)
    plt.close()
