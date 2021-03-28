const allPOS = [
    {
        key: 'ADJ',
        pos: 'adjective'
    },
    {
        key: 'ADP',
        pos: 'adposition'
    },
    {
        key: 'ADV',
        pos: 'adverb'
    },
    {
        key: 'AUX',
        pos: 'auxilliary'
    },
    {
        key: 'CONJ',
        pos: 'conjunction'
    },
    {
        key: 'CCONJ',
        pos: 'coordinating conjunction'
    },
    {
        key: 'DET',
        pos: 'determiner'
    },
    {
        key: 'INTJ',
        pos: 'interjection'
    },
    {
        key: 'NOUN',
        pos: 'noun'
    },
    {
        key: 'NUM',
        pos: 'numerical'
    },
    {
        key: 'PART',
        pos: 'particle'
    },
    {
        key: 'PRON',
        pos: 'pronoun'
    },
    {
        key: 'PROPN',
        pos: 'proper noun'
    },
    {
        key: 'PUNCT',
        pos: 'punctuation'
    },
    {
        key: 'SCONJ',
        pos: 'subordinating conjunction'
    },
    {
        key: 'SYM',
        pos: 'symbol'
    },
    {
        key: 'VERB',
        pos: 'verb'
    },
    {
        key: 'X',
        pos: 'other'
    },
    {
        key: 'SPACE',
        pos: 'space'
    },
]

export default function whatPOS(thekey){
    const pos_obj = allPOS.filter(apos => apos.key === thekey);
    return pos_obj[0].pos
}