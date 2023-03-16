import warnings
import json
import re
import requests
from typing import Any, List
from bs4 import BeautifulSoup

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "根据以下B站视频的标题、简介和字幕，总结摘要：\n"
    "{}\n"
    "\n"
    "\n"
    "摘要：\n"
    )

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def get_cid(cookies, bvid=None, aid=None):
    if bvid:
        info_url = f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}&jsonp=jsonp"
    else:
        info_url = f"https://api.bilibili.com/x/player/pagelist?aid={aid[2:]}&jsonp=jsonp"
    response = requests.get(info_url, headers=headers, cookies=cookies)

    json_data = json.loads(response.text)
    cid = json_data["data"][0]["cid"]
    return cid


def get_subtitle_list(cookies, bvid=None, aid=None):
    cid = get_cid(cookies, bvid, aid)
    url = f'https://api.bilibili.com/x/player/v2?cid={cid}'
    url += f"&aid={aid[2:]}" if aid else f"&bvid={bvid}"
    info = requests.get(url, headers=headers, cookies=cookies)
    subtitles = json.loads(info.text)['data']['subtitle']['subtitles']
    return subtitles


class BilibiliTranscriptReader():
    """Bilibili Transcript and video info reader."""

    @staticmethod
    def get_bilibili_info_and_subs(bili_url, cookies):
        
        bvid = None
        aid = None
        pic = None
        try:
            bvid = re.search(r"BV\w+", bili_url).group()
            url = f'https://www.bilibili.com/video/{bvid}'
        except:
            aid = re.search(r"av\w+", bili_url).group()
            url = f'https://www.bilibili.com/video/{aid}'

        # 发送GET请求，获取网页HTML源码
        response = requests.get(url, headers=headers, cookies=cookies)
        html = response.content.decode()

        # 使用BeautifulSoup库解析HTML源码
        soup = BeautifulSoup(html, 'html.parser')

        # 提取视频标题
        title = soup.find('h1', class_='video-title').text.strip()

        # 提取视频简介
        desc = soup.find('div', class_='desc-info').text.strip()

        # 提取视频字幕
        sub_list = get_subtitle_list(cookies, bvid, aid)
        if sub_list and len(sub_list) > 0:
            sub_url = sub_list[0]["subtitle_url"]
            result = requests.get("https:" + sub_url)
            raw_sub_titles = json.loads(result.content)["body"]
            raw_transcript = " ".join([c["content"] for c in raw_sub_titles])
            # Add basic video info to transcript
            raw_transcript_with_meta_info = f"<视频标题> {title}\n<视频简介> {desc}\n<字幕内容> {raw_transcript}"
            return raw_transcript_with_meta_info, pic, bvid
        else:
            raw_transcript = ""
            warnings.warn(
                f"No subtitles found for video: {bili_url}. Return Empty transcript."
            )
            return raw_transcript, pic, bvid

    def load_data(self, video_urls: str, cookies: str, **load_kwargs: Any) -> str:
        results = ""
        cookies = {i.split("=")[0]:i.split("=")[1] for i in cookies.split(";")}
        pic = None
        bvid = ""
        # try:
        transcript, pic, bvid = self.get_bilibili_info_and_subs(video_urls, cookies)
        if transcript != "":
            results = DEFAULT_SUMMARY_PROMPT_TMPL.format(transcript)
        else:
            return "", pic, bvid
        return results, pic, bvid

if __name__ == '__main__':
    # Test
    cookies = ""
    BTR = BilibiliTranscriptReader()
    BTR.load_data('https://www.bilibili.com/video/av568208936/', cookies=cookies)
