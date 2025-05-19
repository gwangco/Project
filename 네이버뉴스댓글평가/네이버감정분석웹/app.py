# app.py
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from collections import Counter
from transformers import pipeline

app = Flask(__name__)

sentiment_analyzer = None
try:
    # 기본 모델 시도
    sentiment_analyzer = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    print("기본 모델(tabularisai/multilingual-sentiment-analysis) 로드 성공")
except Exception as e:
    print(f"**기본 모델 로드 오류 발생:** {e}")
    print("**대체 모델(koelectra) 로드를 시도합니다...**")
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="monologg/koelectra-base-v2-discriminator")
        print("대체 모델(koelectra) 로드 성공")
    except Exception as e2:
        print(f"**대체 모델(koelectra) 로드 오류 발생:** {e2}")
        print("**감성 분석 모델 로드에 실패했습니다. 일부 기능이 제한될 수 있습니다.**")
        sentiment_analyzer = None # 명시적으로 None 설정

def fetch_naver_comments(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # User-Agent 추가 (선택 사항, 때때로 차단 방지에 도움)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")


    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    comments_list = []
    error = None

    try:
        driver.get(url)
        print(f"URL 접속 시도: {url}")
        time.sleep(3)  # 초기 페이지 로딩 대기

       
        try:
            WebDriverWait(driver, 10).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, "comment_frame"))
            )
            print("comment_frame iframe으로 전환 성공")
            time.sleep(1) 
        except Exception as iframe_e:
            print(f"comment_frame iframe 전환 실패 또는 없음: {iframe_e}")
           
            pass 


        click_count = 0
        max_clicks = 10 # 무한 루프 방지
        while click_count < max_clicks:
            try:
                more_button_selector = ".u_cbox_btn_more"
                # 더보기 버튼이 화면에 나타나고 클릭 가능할 때까지 대기
                more_button = WebDriverWait(driver, 3).until( # 대기 시간 짧게 조정
                    EC.element_to_be_clickable((By.CSS_SELECTOR, more_button_selector))
                )
                # JavaScript로 클릭 시도 (가끔 일반 click()보다 안정적일 수 있음)
                driver.execute_script("arguments[0].click();", more_button)
                print("댓글 더보기 클릭")
                click_count += 1
                time.sleep(1.5)  # 댓글 로딩 대기
            except Exception as more_e:
                print("더 이상 '더보기' 버튼이 없거나 클릭할 수 없습니다.")
                break
        
        
        comment_content_selector = ".u_cbox_contents"
        
        # 댓글 요소들이 나타날 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, comment_content_selector))
        )
        time.sleep(1) # 요소 로드 후 짧은 대기

        comment_elements = driver.find_elements(By.CSS_SELECTOR, comment_content_selector)

        if not comment_elements:
            error = "댓글이 없습니다."
            print("❌ 댓글이 없습니다.")
        else:
            for comment in comment_elements:
                comment_text = comment.text.strip()
                if comment_text: # 빈 댓글은 제외
                    comments_list.append(comment_text)
            print(f"✅ 총 {len(comments_list)}개의 댓글을 수집했습니다.")
            if not comments_list: # 요소는 찾았으나 텍스트가 모두 비어있는 경우
                 error = "댓글 내용은 있으나 텍스트를 가져올 수 없거나 모두 빈 댓글입니다."
                 print("❌ 댓글 내용은 있으나 텍스트를 가져올 수 없거나 모두 빈 댓글입니다.")


    except Exception as e:
        error_message = f"댓글 수집 중 오류 발생: {str(e)}"
        # 스택 트레이스를 포함하지 않고 간단한 오류 메시지만 포함하도록 수정
        error = error_message
        print(f"❌ {error_message}")
    finally:
        driver.quit()

    return comments_list, error

def analyze_sentiment_raw(comment_text):
    """모델로부터 원시 감성 레이블을 반환합니다 ('positive', 'negative', 'neutral' 또는 오류 스트링)."""
    if sentiment_analyzer is None:
        return "모델_로드_실패"
    try:
        # 댓글이 너무 길면 모델 처리 시 오류 발생 가능성 있으므로 적절히 자르기 (예: 512자)
        # 모델의 최대 토큰 길이에 따라 조절 필요
        max_len = 512 
        truncated_comment = comment_text[:max_len]

        result = sentiment_analyzer(truncated_comment)[0]
        label = result['label'].lower() # 소문자로 변환하여 일관성 유지

        if 'positi' in label:
            return 'positive'
        elif 'negati' in label:
            return 'negative'
        elif 'neutra' in label:
            return 'neutral'
        elif label == 'label_0': 
            return 'negative' # 모델에 따라 조정 필요
        elif label == 'label_1': 
            return 'positive' # 모델에 따라 조정 필요
        elif label == 'label_2':
            return 'neutral' # 모델에 따라 조정 필요
        else:
            print(f"인식되지 않는 감성 레이블: {result['label']} for comment: {truncated_comment[:30]}...")
            return "판단_불가" 
    except Exception as e:
        print(f"개별 댓글 감성 분석 오류: {comment_text[:30]}... - {e}")
        return "판단_오류"

def get_korean_sentiment_label(raw_label):
    """원시 감성 레이블을 한국어 레이블로 변환합니다."""
    if raw_label == "positive":
        return "긍정"
    elif raw_label == "negative":
        return "부정"
    elif raw_label == "neutral":
        return "중립"
    elif raw_label == "모델_로드_실패":
        return "모델 로드 실패"
    elif raw_label == "판단_오류":
        return "판단 오류"
    elif raw_label == "판단_불가":
        return "판단 불가"
    else:
        return raw_label 

def process_and_analyze(url):
    comments, fetch_error = fetch_naver_comments(url)
    results = []
    overall_sentiment = "판단 어려움" 

    if fetch_error and not comments: 
        return results, fetch_error, overall_sentiment

    if not comments: 
        error_message = fetch_error if fetch_error else "댓글이 없습니다."
        return results, error_message, "데이터 없음"

    if sentiment_analyzer is None:
        error_message = "감성 분석 모델 로드 실패로 분석을 수행할 수 없습니다."
        if fetch_error:
            error_message = f"{fetch_error}\n{error_message}"
        for comment_text in comments:
            results.append({'comment': comment_text, 'sentiment': '분석 불가'})
        return results, error_message, "분석 불가"

    raw_sentiments = [analyze_sentiment_raw(comment) for comment in comments]
    
    for i, comment_text in enumerate(comments):
        korean_label = get_korean_sentiment_label(raw_sentiments[i])
        results.append({'comment': comment_text, 'sentiment': korean_label})

    sentiment_counts = Counter(raw_sentiments)
    positive_count = sentiment_counts.get("positive", 0)
    negative_count = sentiment_counts.get("negative", 0)
    neutral_count = sentiment_counts.get("neutral", 0)
    
    analyzable_comments_count = positive_count + negative_count + neutral_count

    if analyzable_comments_count > 0:
        if positive_count > negative_count and positive_count >= neutral_count:
            overall_sentiment = "긍정"
        elif negative_count > positive_count and negative_count >= neutral_count:
            overall_sentiment = "부정"
        elif neutral_count > positive_count and neutral_count > negative_count: 
            overall_sentiment = "중립"
        elif positive_count > negative_count : 
            overall_sentiment = "긍정"
        elif negative_count > positive_count : 
            overall_sentiment = "부정"
        else: 
            overall_sentiment = "중립"
            
    elif len(comments) > 0 and analyzable_comments_count == 0: 
        overall_sentiment = "분석 불가"

    return results, fetch_error, overall_sentiment


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        naver_url = request.form.get('naver_url')
        if not naver_url:
            return render_template('index.html', error="URL을 입력해주세요.")

        analysis_results, error_msg, overall_sentiment_result = process_and_analyze(naver_url)

        if error_msg and not analysis_results and overall_sentiment_result != "데이터 없음": # 데이터 없음은 댓글이 없는 정상 케이스
             # "댓글이 없습니다." 메시지는 index.html이 아닌 result.html에서 보여주는 것이 더 적절할 수 있음
            if error_msg == "댓글이 없습니다.":
                 return render_template('result.html', 
                               url=naver_url, 
                               results=analysis_results, 
                               overall_sentiment="데이터 없음", # 댓글이 없음을 명확히
                               error_message=error_msg)
            return render_template('index.html', error=error_msg)
        
        return render_template('result.html', 
                               url=naver_url, 
                               results=analysis_results, 
                               overall_sentiment=overall_sentiment_result,
                               error_message=error_msg)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
