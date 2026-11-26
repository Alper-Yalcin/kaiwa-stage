// scripts/build_web.py üretti; elle düzenleme.
window.CHAT_DATA = {
 "characters": [
  {
   "name": "yuki",
   "description": "JVNV F1 — genç kadın, sıcak ve ifadeli (duygu stilleri güçlü)"
  },
  {
   "name": "hana",
   "description": "JVNV F2 — kadın, net ve dramatik"
  },
  {
   "name": "kenta",
   "description": "JVNV M1 — genç erkek, canlı"
  },
  {
   "name": "takeshi",
   "description": "JVNV M2 — erkek, pes ve güçlü"
  }
 ],
 "turns": [
  {
   "id": "t01",
   "user": {
    "ja": "こんにちは！今日はいい天気ですね。",
    "tr": "Merhaba! Bugün hava çok güzel, değil mi?",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "konnichiwa! kyou wa ii tenki desu ne."
   },
   "reply": {
    "ja": "こんにちは！本当にいい天気ですね。散歩に行きたくなります。",
    "tr": "Merhaba! Gerçekten güzel. İnsanın yürüyüşe çıkası geliyor.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "konnichiwa! hontouni ii tenki desu ne. sanpo ni ikitaku narimasu."
   }
  },
  {
   "id": "t02",
   "user": {
    "ja": "お名前は何ですか？",
    "tr": "Adınız ne?",
    "romaji": "o namae wa nan desu ka?"
   },
   "reply": {
    "ja": "私の名前はユキです。よろしくお願いします。",
    "tr": "Benim adım Yuki. Memnun oldum.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "watashi no namae wa yuki desu. yoroshikuonegaishimasu."
   }
  },
  {
   "id": "t03",
   "user": {
    "ja": "日本語を勉強しています。",
    "tr": "Japonca çalışıyorum.",
    "romaji": "nihongo o benkyou shite imasu."
   },
   "reply": {
    "ja": "すごいですね！どのくらい勉強していますか？",
    "tr": "Harika! Ne kadar zamandır çalışıyorsunuz?",
    "style": "surprise",
    "style_weight": 0.8,
    "romaji": "sugoi desu ne! dono kurai benkyou shite imasu ka?"
   }
  },
  {
   "id": "t04",
   "user": {
    "ja": "週末は何をする予定ですか？",
    "tr": "Hafta sonu ne yapmayı planlıyorsunuz?",
    "romaji": "shuumatsu wa nani o suru yotei desu ka?"
   },
   "reply": {
    "ja": "友達と映画を見に行く予定です。あなたは？",
    "tr": "Arkadaşlarla sinemaya gitmeyi planlıyorum. Ya siz?",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "tomodachi to eiga o mi ni iku yotei desu. anata wa?"
   }
  },
  {
   "id": "t05",
   "user": {
    "ja": "お腹が空きました。何か食べたいです。",
    "tr": "Acıktım. Bir şeyler yemek istiyorum.",
    "style": "sad",
    "style_weight": 0.6,
    "romaji": "onaka ga akimashita. nani ka tabetaidesu."
   },
   "reply": {
    "ja": "駅前に美味しいラーメン屋がありますよ。一緒に行きましょう。",
    "tr": "İstasyonun önünde güzel bir ramen dükkânı var. Birlikte gidelim.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "eki mae ni oishii raamen ya ga arimasu yo. issho ni ikimashou."
   }
  },
  {
   "id": "t06",
   "user": {
    "ja": "今何時ですか？",
    "tr": "Saat kaç?",
    "romaji": "ima nan ji desu ka?"
   },
   "reply": {
    "ja": "今は午後三時半です。",
    "tr": "Şu an öğleden sonra üç buçuk.",
    "romaji": "ima wa gogo san jihan desu."
   }
  },
  {
   "id": "t07",
   "user": {
    "ja": "明日の天気はどうですか？",
    "tr": "Yarın hava nasıl olacak?",
    "romaji": "ashita no tenki wa dou desu ka?"
   },
   "reply": {
    "ja": "明日は雨が降るそうです。傘を忘れないでくださいね。",
    "tr": "Yarın yağmur yağacakmış. Şemsiyenizi unutmayın.",
    "romaji": "ashita wa ame ga furu sou desu. kasa o wasurenaide kudasai ne."
   }
  },
  {
   "id": "t08",
   "user": {
    "ja": "趣味は何ですか？",
    "tr": "Hobiniz ne?",
    "romaji": "shumi wa nan desu ka?"
   },
   "reply": {
    "ja": "音楽を聴くことと、ゲームをすることです。",
    "tr": "Müzik dinlemek ve oyun oynamak.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "ongaku o kiku koto to, geimu o suru koto desu."
   }
  },
  {
   "id": "t09",
   "user": {
    "ja": "好きな食べ物は何ですか？",
    "tr": "Sevdiğiniz yemek ne?",
    "romaji": "suki na tabemono wa nan desu ka?"
   },
   "reply": {
    "ja": "寿司が大好きです。特にサーモンが好きです。",
    "tr": "Suşiye bayılırım. Özellikle somon.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "sushi ga daisuki desu. tokuni saamon ga suki desu."
   }
  },
  {
   "id": "t10",
   "user": {
    "ja": "ちょっと疲れました。",
    "tr": "Biraz yoruldum.",
    "style": "sad",
    "style_weight": 0.6,
    "romaji": "chotto tsukaremashita."
   },
   "reply": {
    "ja": "お疲れ様です。少し休んだほうがいいですよ。",
    "tr": "Kolay gelsin. Biraz dinlenseniz iyi olur.",
    "romaji": "otsukaresama desu. sukoshi yasunda hou ga ii desu yo."
   }
  },
  {
   "id": "t11",
   "user": {
    "ja": "ありがとうございました。",
    "tr": "Teşekkür ederim.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "arigatou gozaimashita."
   },
   "reply": {
    "ja": "どういたしまして。また話しましょう。",
    "tr": "Rica ederim. Yine konuşalım.",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "dou itashimashite. mata hanashimashou."
   }
  },
  {
   "id": "t12",
   "user": {
    "ja": "さようなら！",
    "tr": "Hoşça kalın!",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "sayounara!"
   },
   "reply": {
    "ja": "さようなら！良い一日を！",
    "tr": "Hoşça kalın! İyi günler!",
    "style": "happy",
    "style_weight": 0.7,
    "romaji": "sayounara! yoi ichinichi o!"
   }
  }
 ],
 "audio_pattern": "audio/{character}/{turn}_{side}.mp3"
};
