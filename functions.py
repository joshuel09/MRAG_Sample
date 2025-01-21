from bs4 import BeautifulSoup
import os
import requests
import json
from transformers import CLIPProcessor, CLIPTextModelWithProjection
import torch

def split_text_with_overlap(text, max_length=256, overlap_percentage=0.25):
        """テキストをオーバーラップ付きでチャンクに分割"""
        if len(text) <= max_length:
            return [text]
        
        overlap_size = int(max_length * overlap_percentage)
        chunks = []
        
        # 潜在的なチャンクの総数を計算
        chunk_starts = range(0, len(text), max_length - overlap_size)
        
        for start in chunk_starts:
            # 最大長または残りのテキストを取得
            chunk = text[start:start + max_length]
            
            # 最後のチャンク以外はスペースで分割を試みる
            if start + max_length < len(text):
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    chunk = chunk[:last_space]
            
            chunks.append(chunk.strip())
        
        return chunks

def parse_html_content_fine_grained(html_content):
    """
    最後のチャンク以外はスペースで分割を試みる
    """
    # HTMLを解析
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 記事タイトルを取得
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"
    
    # 初期化
    structured_content = []
    current_section = "Main"  # セクションが見つからない場合のデフォル
    
    # 見出しと段落を検索
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'p'])
    
    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name == 'p' and element.get_text().strip():

            text = element.get_text().strip()
            # オーバーラップ付きでテキストをチャンク化
            text_chunks = split_text_with_overlap(text)
            
            for chunk in text_chunks:
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'text': chunk
                })
    
    return structured_content

def parse_html_content(html_content):
    """
    HTMLコンテンツを解析し、画像とキャプションを抽出する
    """
    # HTMLを解析
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 記事タイトルを取得
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"
    
    # 初期化
    structured_content = []
    current_section = "Main"  # セクションが見つからない場合のデフォルト
    
    # 見出しと画像を検索
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol'])
    
    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name in ['p', 'ul', 'ol']:
            text = element.get_text().strip()
            # 非空で30文字以上のテキストのみ追加
            if text and len(text) >= 30:
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'text': text
                })
    
    return structured_content

def parse_html_images(html_content):
    """
    HTMLコンテンツを解析し、画像とキャプションを抽出
    """
    # HTMLを解析
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 記事タイトルを取得
    article_title = soup.find('title').get_text().strip() if soup.find('title') else "Untitled"
    
    # 初期化
    structured_content = []
    current_section = "Main"  # セクションが見つからない場合のデフォルト
    
    # 見出しと画像を検索
    content_elements = soup.find_all(['h1', 'h2', 'h3', 'img', 'figure'])
    
    for element in content_elements:
        if element.name in ['h1', 'h2', 'h3']:
            current_section = element.get_text().strip()
        elif element.name == 'img':
            # 画像パスを取得
            image_url = element.get('src', '')
            
            if image_url:  # 有効な画像URLがある場合のみ処理
                # 画像をダウンロード
                response = requests.get(image_url)
                if response.status_code == 200:
                    # ディレクトリを作成
                    os.makedirs('images', exist_ok=True)
                    
                    # 画像ファイル名を取得
                    image_filename = os.path.basename(image_url)
                    if "." not in image_filename:
                        image_filename = f"{image_filename}.jpg"
                    
                    # ローカルパスを定義
                    local_image_path = os.path.join('images', image_filename)
                    
                    # Save the image to the local file path
                    with open(local_image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # 画像を保存
                    image_path = local_image_path
                else:
                    image_path = ''
            
            # キャプションを取得
            caption = element.get('alt', '')
            if not caption and element.parent.name == 'figure':
                figcaption = element.parent.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text().strip()
            
            if image_path:  # 有効な画像パスがある場合のみ追加
                structured_content.append({
                    'article_title': article_title,
                    'section': current_section,
                    'image_path': image_path,
                    'caption': caption or "No caption available"
                })
    
    return structured_content

def save_to_json(structured_content, output_file='output.json'):
    """
    構造化データをJSONファイルに保存
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_content, f, indent=4, ensure_ascii=False)

def load_from_json(input_file):
    """
    JSONファイルから構造化データを読み込む
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def embed_text(text):
    """
    CLIPを使用してテキストを埋め込みベクトルに変換
    """
    
    # モデルをインポート
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    # プロセッサをインポート（テキストのトークン化と画像の前処理を担当）
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # テキストと画像を前処理
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    # CLIPを使用して埋め込みを計算
    outputs = model(**inputs)

    return outputs.text_embeds

def similarity_search(query_embed, target_embeddings, content_list, k=5, threshold=0.05, temperature=0.5):
    """
    埋め込みベクトルを使用して類似度検索を実行
    """
    # 類似度を計算
    similarities = torch.matmul(query_embed, target_embeddings.T)
    
    # ソフトマックスで類似度を再スケーリング
    scores = torch.nn.functional.softmax(similarities/temperature, dim=1)
    
    # ソートされたインデックスとスコアを取得
    sorted_indices = scores.argsort(descending=True)[0]
    sorted_scores = scores[0][sorted_indices]
    
    # 閾値でフィルタリングし、上位k件を取得
    filtered_indices = [
        idx.item() for idx, score in zip(sorted_indices, sorted_scores) 
        if score.item() >= threshold
    ][:k]
    
    # 対応するコンテンツ項目とスコアを取得
    top_results = [content_list[i] for i in filtered_indices]
    result_scores = [scores[0][i].item() for i in filtered_indices]
    
    return top_results, result_scores

def construct_prompt(query, text_results, image_results):
    """
    LLMにレスポンスを生成させるためのプロンプトを構築
    """

    text_context = ""
    for text in text_results:
        if text_results:
            text_context = text_context + "**記事タイトル:** " + text['article_title'] + "\n"
            text_context = text_context + "**セクション:**  " + text['section'] + "\n"
            text_context = text_context + "**スニペット:** " + text['text'] + "\n\n"

    image_context = ""
    for image in image_results:
        if image_results:
            image_context = image_context + "**記事タイトル:** " + image['article_title'] + "\n"
            image_context = image_context + "**セクション:**  " + image['section'] + "\n"
            image_context = image_context + "**画像パス:**  " + image['image_path'] + "\n"
            image_context = image_context + "**キャプション:** " + image['caption'] + "\n\n"

    # 類似度検索を実行
    return f"""以下の質問「{query}」と、以下の関連スニペットが提供されています:

    {text_context}
    {image_context}

    提供されたスニペットの情報を基に、簡潔かつ正確な回答を作成してください。

    """


def context_retrieval(query, text_embeddings, image_embeddings, text_content_list, image_content_list, 
                    text_k=15, image_k=5, 
                    text_threshold=0.01, image_threshold=0.25,
                    text_temperature=0.25, image_temperature=0.5):
    """
    埋め込みベクトルを使用してコンテキスト検索を実行し、上位k件の結果を返す
    """
    # CLIPを使用してクエリを埋め込み
    query_embed = embed_text(query)

    # 類似度検索を実行
    text_results, _ = similarity_search(query_embed, text_embeddings, text_content_list, k=text_k, threshold=text_threshold, temperature=text_temperature)
    image_results, _ = similarity_search(query_embed, image_embeddings, image_content_list, k=image_k, threshold=image_threshold, temperature=image_temperature)

    return text_results, image_results