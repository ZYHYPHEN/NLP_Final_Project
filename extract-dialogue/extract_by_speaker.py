#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 jsonl 文件按角色提取对话（兼容多种常见对话格式）

用法示例:
    python extract_by_speaker.py -s "角色名" -i xyj_dialogues_concurrent.jsonl

输出默认打印到 stdout，也可通过 -o 写入 jsonl 文件。
"""
import json
import argparse
from pathlib import Path
from typing import Any, Dict


def iter_turns(obj):
    if isinstance(obj, list):
        for t in obj:
            yield t
        return
    if isinstance(obj, dict):
        for key in ('utterances', 'dialogue', 'turns', 'conversations', 'messages', '对话'):
            if key in obj and isinstance(obj[key], list):
                for t in obj[key]:
                    yield t
                return
    # fallback: 找到第一个列表值
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                for t in v:
                    yield t
                return


def get_speaker_and_text(turn: Any):
    if isinstance(turn, str):
        return None, turn
    if not isinstance(turn, dict):
        return None, None

    # 尝试常见的说话者字段
    speaker = None
    for sk in ('speaker', 'speaker_id', 'spk', 'role', '角色', '说话者', 'name', 'from', 'sender'):
        if sk in turn:
            speaker = turn.get(sk)
            break

    # 常见文本字段（优先包含 dialogue/台词/发言 等）
    text = None
    for tk in ('dialogue', '台词', '发言', 'text', 'utterance', 'content', '内容', 'message', '说话'):
        if tk in turn:
            text = turn.get(tk)
            break

    # 如果仍无文本，尝试取第一个字符串值
    if text is None:
        for v in turn.values():
            if isinstance(v, str):
                text = v
                break

    if speaker is not None:
        speaker = str(speaker)
    if text is not None:
        text = str(text)

    return speaker, text


def matches(speaker: str, target: str) -> bool:
    if speaker is None:
        return False
    s = speaker.lower().strip()
    t = target.lower().strip()
    return t in s or s in t


def extract(input_path: Path, speaker: str):
    results = []
    with input_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            hits = []
            # 直接顶层可能就是一个说话项
            spk, txt = get_speaker_and_text(obj)
            if spk and txt and matches(spk, speaker):
                hits.append(txt)

            # 遍历可能的turns
            for turn in iter_turns(obj):
                spk, txt = get_speaker_and_text(turn)
                if spk and txt and matches(spk, speaker):
                    hits.append(txt)

            if hits:
                results.append({
                    'dialogue_index': idx,
                    'speaker': speaker,
                    'hits': hits,
                })

    return results


def extract_replies(input_path: Path, speaker: str, include_prev_speaker: bool = False):
    """按文件行顺序提取目标角色对前一轮说话者的回复。

    输出格式为列表，每项为 {"instruction": ..., "input":"", "output": ...}
    """
    objs = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            objs.append(obj)

    pairs = []
    for i in range(1, len(objs)):
        prev = objs[i - 1]
        cur = objs[i]

        prev_spk, prev_txt = get_speaker_and_text(prev)
        cur_spk, cur_txt = get_speaker_and_text(cur)

        if cur_txt is None or cur_spk is None or prev_txt is None:
            continue

        if matches(cur_spk, speaker) and (not matches(prev_spk, speaker)):
            if include_prev_speaker and prev_spk:
                instruction = f"{prev_spk}: {prev_txt}"
            else:
                instruction = prev_txt

            pairs.append({
                "instruction": instruction,
                "input": "",
                "output": cur_txt
            })

    return pairs


def extract_replies_multiple(input_path: Path, speakers, include_prev_speaker: bool = False):
    """为多个目标角色提取对前一轮说话者的回复，返回包含 speaker 字段的记录列表。"""
    objs = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            objs.append(obj)

    pairs = []
    for i in range(1, len(objs)):
        prev = objs[i - 1]
        cur = objs[i]

        prev_spk, prev_txt = get_speaker_and_text(prev)
        cur_spk, cur_txt = get_speaker_and_text(cur)

        if cur_txt is None or cur_spk is None or prev_txt is None:
            continue

        for target in speakers:
            if matches(cur_spk, target) and (not matches(prev_spk, target)):
                if include_prev_speaker and prev_spk:
                    instruction = f"{prev_spk}: {prev_txt}"
                else:
                    instruction = prev_txt

                pairs.append({
                    "speaker": cur_spk,
                    "instruction": instruction,
                    "input": "",
                    "output": cur_txt,
                    "dialogue_index": i
                })
                break

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='xyj_dialogues_concurrent.jsonl')
    parser.add_argument('-s', '--speaker', help='要查找的角色名（支持部分匹配）')
    parser.add_argument('--speakers', help='多个角色名，逗号分隔。例如：悟空,猴王')
    parser.add_argument('--replies', action='store_true', help='提取目标角色对前一轮说话者的回复，输出为 instruction/input/output 格式')
    parser.add_argument('--include-prev-speaker', action='store_true', help='在 instruction 中包含前一轮说话者的名字')
    parser.add_argument('-o', '--output', help='输出jsonl文件，若不提供则打印到stdout')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"输入文件不存在: {inp}")

    # 解析要查询的角色集合
    target_speakers = []
    if args.speakers:
        target_speakers = [s.strip() for s in args.speakers.split(',') if s.strip()]
    elif args.speaker:
        target_speakers = [args.speaker]
    else:
        raise SystemExit('请通过 -s 或 --speakers 指定要查询的角色')

    if args.replies:
        results = extract_replies_multiple(inp, target_speakers, include_prev_speaker=args.include_prev_speaker)
        if args.output:
            outp = Path(args.output)
            with outp.open('w', encoding='utf-8') as fo:
                fo.write(json.dumps(results, ensure_ascii=False, indent=4))
            print(f'Wrote {len(results)} reply records to {outp}')
        else:
            print(json.dumps(results, ensure_ascii=False, indent=4))
    else:
        # 如果非 replies 模式，仍支持多个角色：分别列出每个角色的 hits
        all_results = []
        for spk in target_speakers:
            res = extract(inp, spk)
            all_results.append({"speaker": spk, "results": res})
        results = all_results
        if args.output:
            outp = Path(args.output)
            with outp.open('w', encoding='utf-8') as fo:
                fo.write(json.dumps(results, ensure_ascii=False, indent=4))
            print(f'Wrote results for {len(target_speakers)} speakers to {outp}')
        else:
            for block in results:
                print('=== Speaker:', block['speaker'], '===')
                for r in block['results']:
                    print('--- Dialogue', r['dialogue_index'], '---')
                    for h in r['hits']:
                        print(h)


if __name__ == '__main__':
    main()
