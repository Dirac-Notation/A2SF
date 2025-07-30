import json
import random
import os
from pathlib import Path
from datetime import datetime

def convert_longbench_to_cnn_format():
    # longbench 디렉토리 경로
    longbench_dir = Path("result_txt/longbench")
    
    # 출력 파일 경로 - 별도 디렉토리에 저장
    output_dir = Path("datasets/converted_longbench")
    output_dir.mkdir(exist_ok=True)
    
    # 타임스탬프를 포함한 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = output_dir / f"longbench_to_cnn_{timestamp}"
    timestamp_dir.mkdir(exist_ok=True)
    
    # longbench 파일들 목록
    longbench_files = list(longbench_dir.glob("*.jsonl"))
    
    if not longbench_files:
        print("LongBench 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 LongBench 파일들: {[f.name for f in longbench_files]}")
    print(f"출력 디렉토리: {timestamp_dir}")
    
    total_converted = 0
    
    for file_path in longbench_files:
        print(f"처리 중: {file_path.name}")
        
        # 현재 파일의 데이터 수집
        file_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            file_data.append(data)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"파일 {file_path.name} 처리 중 오류: {e}")
            continue
        
        print(f"  - {file_path.name}: {len(file_data)}개 데이터 발견")
        
        if len(file_data) < 5:
            print(f"  - {file_path.name}: 데이터가 5개 미만입니다. 모든 데이터를 사용합니다.")
            selected_data = file_data
        else:
            # 랜덤으로 5개 선택
            selected_data = random.sample(file_data, 5)
        
        # CNN DailyMail 포맷으로 변환
        converted_data = []
        for item in selected_data:
            # LongBench 데이터에서 필요한 필드 추출
            # LongBench 포맷: {"input_prompt": "...", "input": "...", "output": "..."}
            # CNN DailyMail 포맷: {"article": "...", "summary_gt": "...", "input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
            
            # input_prompt와 input을 결합하여 article 생성
            article = ""
            if 'input_prompt' in item:
                article += item['input_prompt']
            if 'input' in item:
                if article:
                    article += "\n\n"
                article += item['input']
            
            # output을 summary_gt로 사용
            summary_gt = item.get('output', '')
            
            # 토큰 수 계산 (간단한 추정)
            input_tokens = len(article.split()) if article else 0
            output_tokens = len(summary_gt.split()) if summary_gt else 0
            total_tokens = input_tokens + output_tokens
            
            # 변환된 데이터 생성
            converted_item = {
                "article": article,
                "summary_gt": summary_gt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "source_file": file_path.name  # 원본 파일 정보 추가
            }
            
            converted_data.append(converted_item)
        
        # 파일명 생성 (원본 파일명에서 확장자 제거하고 _5samples 추가)
        original_name = file_path.stem  # 확장자 제거
        output_filename = f"{original_name}_5samples.jsonl"
        output_file = timestamp_dir / output_filename
        
        # 결과를 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  - 저장 완료: {output_filename} ({len(converted_data)}개)")
        total_converted += len(converted_data)
    
    print(f"\n총 {total_converted}개의 데이터를 변환했습니다.")
    print(f"모든 파일이 {timestamp_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    convert_longbench_to_cnn_format() 