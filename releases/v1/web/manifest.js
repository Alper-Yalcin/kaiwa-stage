// scripts/build_web.py üretti; elle düzenleme.
window.CHAT_DATA = {
 "characters": [
  {
   "name": "deep_narrator",
   "description": "Pes, düz, anlatıcı erkek sesi; yavaş ve kararlı"
  },
  {
   "name": "chill_male",
   "description": "Rahat, ağır konuşan erkek; düşük perde"
  },
  {
   "name": "energetic_male",
   "description": "Hızlı, enerjik orta perdeli erkek"
  },
  {
   "name": "young_male",
   "description": "Genç, ifadeli, yüksek perdeli erkek"
  },
  {
   "name": "calm_female",
   "description": "Sakin, alçak perdeli kadın; anons/rehber tonu"
  },
  {
   "name": "bright_female",
   "description": "Canlı, hızlı konuşan kadın"
  },
  {
   "name": "excited_female",
   "description": "Tiz, heyecanlı, geniş tonlamalı kadın"
  },
  {
   "name": "mysterious_voice",
   "description": "Fısıltılı, yavaş, gizemli kadın sesi"
  },
  {
   "name": "whisper_male",
   "description": "Fısıltılı erkek; sır verir gibi"
  },
  {
   "name": "playful_falsetto",
   "description": "Falsetto/kafa sesi (~500 Hz), oyuncu ve tiz kadın karakteri"
  }
 ],
 "turns": [
  {
   "id": "t01",
   "user": {
    "ja": "こんにちは！今日はいい天気ですね。",
    "tr": "Merhaba! Bugün hava çok güzel, değil mi?"
   },
   "reply": {
    "ja": "こんにちは！本当にいい天気ですね。散歩に行きたくなります。",
    "tr": "Merhaba! Gerçekten güzel. İnsanın yürüyüşe çıkası geliyor."
   }
  },
  {
   "id": "t02",
   "user": {
    "ja": "お名前は何ですか？",
    "tr": "Adınız ne?"
   },
   "reply": {
    "ja": "私の名前はユキです。よろしくお願いします。",
    "tr": "Benim adım Yuki. Memnun oldum."
   }
  },
  {
   "id": "t03",
   "user": {
    "ja": "日本語を勉強しています。",
    "tr": "Japonca çalışıyorum."
   },
   "reply": {
    "ja": "すごいですね！どのくらい勉強していますか？",
    "tr": "Harika! Ne kadar zamandır çalışıyorsunuz?"
   }
  },
  {
   "id": "t04",
   "user": {
    "ja": "週末は何をする予定ですか？",
    "tr": "Hafta sonu ne yapmayı planlıyorsunuz?"
   },
   "reply": {
    "ja": "友達と映画を見に行く予定です。あなたは？",
    "tr": "Arkadaşlarla sinemaya gitmeyi planlıyorum. Ya siz?"
   }
  },
  {
   "id": "t05",
   "user": {
    "ja": "お腹が空きました。何か食べたいです。",
    "tr": "Acıktım. Bir şeyler yemek istiyorum."
   },
   "reply": {
    "ja": "駅前に美味しいラーメン屋がありますよ。一緒に行きましょう。",
    "tr": "İstasyonun önünde güzel bir ramen dükkânı var. Birlikte gidelim."
   }
  },
  {
   "id": "t06",
   "user": {
    "ja": "今何時ですか？",
    "tr": "Saat kaç?"
   },
   "reply": {
    "ja": "今は午後三時半です。",
    "tr": "Şu an öğleden sonra üç buçuk."
   }
  },
  {
   "id": "t07",
   "user": {
    "ja": "明日の天気はどうですか？",
    "tr": "Yarın hava nasıl olacak?"
   },
   "reply": {
    "ja": "明日は雨が降るそうです。傘を忘れないでくださいね。",
    "tr": "Yarın yağmur yağacakmış. Şemsiyenizi unutmayın."
   }
  },
  {
   "id": "t08",
   "user": {
    "ja": "趣味は何ですか？",
    "tr": "Hobiniz ne?"
   },
   "reply": {
    "ja": "音楽を聴くことと、ゲームをすることです。",
    "tr": "Müzik dinlemek ve oyun oynamak."
   }
  },
  {
   "id": "t09",
   "user": {
    "ja": "好きな食べ物は何ですか？",
    "tr": "Sevdiğiniz yemek ne?"
   },
   "reply": {
    "ja": "寿司が大好きです。特にサーモンが好きです。",
    "tr": "Suşiye bayılırım. Özellikle somon."
   }
  },
  {
   "id": "t10",
   "user": {
    "ja": "ちょっと疲れました。",
    "tr": "Biraz yoruldum."
   },
   "reply": {
    "ja": "お疲れ様です。少し休んだほうがいいですよ。",
    "tr": "Kolay gelsin. Biraz dinlenseniz iyi olur."
   }
  },
  {
   "id": "t11",
   "user": {
    "ja": "ありがとうございました。",
    "tr": "Teşekkür ederim."
   },
   "reply": {
    "ja": "どういたしまして。また話しましょう。",
    "tr": "Rica ederim. Yine konuşalım."
   }
  },
  {
   "id": "t12",
   "user": {
    "ja": "さようなら！",
    "tr": "Hoşça kalın!"
   },
   "reply": {
    "ja": "さようなら！良い一日を！",
    "tr": "Hoşça kalın! İyi günler!"
   }
  }
 ],
 "audio_pattern": "audio/{character}/{turn}_{side}.mp3"
};
