"""一键设置 API key：在终端粘贴 key 回车即可，不用记事本。

用法：
  .venv\\Scripts\\python setkey.py
"""
import re
import sys
from getpass import getpass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ENV = Path(__file__).resolve().parent / ".env"


def main():
    key = getpass("粘贴你的 API key 后回车（输入不会显示）: ").strip().strip('"').strip("'")
    if not key:
        sys.exit("没有输入任何内容。")
    if len(key) < 16 or key.count(".") != 1:
        sys.exit(f"这不像一个智谱 key（应形如 xxxxxxxx.xxxxxxxx，你输入的长度 {len(key)}）。"
                 "请回控制台重新复制完整 key。")
    text = ENV.read_text(encoding="utf-8")
    new_text, n = re.subn(r"(?m)^LLM_API_KEY=.*$", f"LLM_API_KEY={key}", text)
    if n != 1:
        sys.exit(".env 里没找到 LLM_API_KEY= 这一行，文件可能被改坏了。")
    ENV.write_text(new_text, encoding="utf-8")
    print(f"已写入 {ENV}")
    print(f"校验：key 长度 {len(key)}，格式看起来正确。现在可以运行 python ingest.py --contextual 了。")


if __name__ == "__main__":
    main()
