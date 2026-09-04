import os
import sys

# 启动 PyQt5 应用程序
if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    os.system(f'python "{app_path}"')
