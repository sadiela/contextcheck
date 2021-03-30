const SUPPORTED_URLS = [
    "cnn.com",
    "foxnews.com",
    "huffpost.com",
    "slate.com",
    "reuters.com",
    "progressive.org",
    "politico.com",
    "theguardian.com",
    "apnews.com",
    "cbsnews.com",
    "cnbc.com",
]

export default function isItSupported(url){
    // const isSupported = SUPPORTED_URLS.some(supported => {
    //     url.includes(supported);
    // });
    var isurl = true
    try {
        isurl = new URL(url);
        isurl = true;
    } catch (_) {
        isurl = false;
    }
    return (isurl);
}