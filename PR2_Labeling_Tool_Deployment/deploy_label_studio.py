import subprocess

def run_label_studio():
    subprocess.run(["label-studio", "start"])

if __name__ == "__main__":
    run_label_studio()