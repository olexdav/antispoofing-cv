import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DEVELOPER_KEY = 'AIzaSyCSqJLGonG-2CicXEtiTkT2okdTaFr9TtY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

queries_file = "/queries1.txt"


def get_queries_set(path):
    queries = set()

    with open(path, 'r') as f:
        for line in f:
            queries.add(line.strip())

    return queries


def parse(cache, ids_amount_chache, next_page="", query=""):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Call the search.list method to retrieve results matching the specified
    # query term.
    print("parseBegin", next_page)
    search_response = youtube.search().list(
        q=query,
        type="video",
        part='id,snippet',
        maxResults=50,
        pageToken=next_page,
        videoDuration='medium',
        videoDefinition='high',
    ).execute()

    video_ids = set()
    i = 0
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        if video_id not in cache:
            cache.add(video_id)
            video_ids.add(video_id)
            ids_amount_cache[query] += 1
            print(video_id, i)
            i += 1

    next_page = search_response.get("nextPageToken", "")
    print("Parse end", next_page)
    add_links_to_file(video_ids)

    return next_page


def add_links_to_file(ids):
    output_index = queries_file[-5]
    with open("video_ids{}.txt".format(output_index), 'a') as f:
            for v_id in ids:
                f.write(v_id + "\n")


if __name__ == '__main__':
    file_path = os.path.dirname(__file__) + queries_file
    cache = set()
    queries = get_queries_set(file_path)

    nextLink = ""
    ids_amount_cache = dict()

    for query in queries:
        ids_amount_cache[query] = 0
        for i in range(3):
            print(query, i)
            while True:
                try:
                    nextLink = parse(cache, ids_amount_cache, nextLink, query)
                    break
                except HttpError as e:
                    print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))
        print("query: ", query, "ID'S AMOUNT: ", ids_amount_cache[query], "---")

    print(ids_amount_cache)


