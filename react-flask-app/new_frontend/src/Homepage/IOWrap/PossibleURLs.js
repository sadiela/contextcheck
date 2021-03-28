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
    const isSupported = SUPPORTED_URLS.some(supported => {
        url.includes(supported);
    });
    var isiturl;
    try {
        const isurl = new URL(url);
        isiturl = true;
    } catch (_) {
        isiturl = false;
    }
    return (isSupported && isiturl);
}