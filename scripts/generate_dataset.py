"""
Step 1: Generate Roast Preference Dataset

Creates three datasets:
1. SFT dataset: (prompt, roast) pairs for supervised fine-tuning
2. Preference dataset: (prompt, chosen, rejected) triples for reward model training
3. PPO prompts: prompts for PPO training

The "chosen" roasts are witty/clever, while "rejected" roasts are mean/lazy/offensive.
This simulates human preference data for alignment.
"""

import json
import os
import random

import pandas as pd
from datasets import Dataset

# ============================================================
# Roast Dataset: Trait prompts + witty (chosen) vs mean (rejected)
# ============================================================

TRAITS = [
    # --- Original 50 ---
    "always late to meetings",
    "obsessed with crypto",
    "talks too much about their cat",
    "thinks they're a LinkedIn influencer",
    "eats loud snacks during Zoom calls",
    "uses too many exclamation marks in emails",
    "peaked in high school",
    "has a podcast nobody listens to",
    "brags about waking up at 5am",
    "still uses Internet Explorer",
    "puts pineapple on pizza",
    "replies all to every email",
    "takes gym selfies constantly",
    "uses Comic Sans unironically",
    "has 47 unread notifications",
    "thinks they're a wine expert",
    "always one-ups everyone's stories",
    "sends voice messages instead of texts",
    "has a man bun",
    "is a self-proclaimed foodie",
    "wears sunglasses indoors",
    "still quotes Borat",
    "has a motivational quote as their wallpaper",
    "overuses the word 'synergy'",
    "drives a Prius and won't shut up about it",
    "brings a guitar to parties",
    "has 12 streaming subscriptions",
    "posts inspirational sunrise photos",
    "microwaves fish in the office",
    "types with two fingers",
    "says 'let's circle back' in every meeting",
    "has a ring light for casual video calls",
    "collects Funko Pops",
    "is always on speakerphone in public",
    "wears a Patagonia vest to every occasion",
    "uses hashtags in real conversation",
    "has read zero books this year but has opinions",
    "treats fantasy football like a real job",
    "brings their own pillow on flights",
    "still uses a flip phone by choice",
    "has strong opinions about fonts",
    "plays devil's advocate constantly",
    "always suggests hiking as a hangout",
    "puts their Venmo in their dating profile",
    "writes 'sent from my iPhone' manually",
    "has a standing desk and tells everyone",
    "uses 'per my last email' aggressively",
    "brings kale chips to potlucks",
    "has an emotional support water bottle",
    "gives unsolicited book recommendations",
    "has no hair",
    # --- New traits: tech & work ---
    "has 200 browser tabs open at all times",
    "uses AI to write every email",
    "says 'let me share my screen' then can't find the window",
    "has a custom mechanical keyboard that sounds like a typewriter",
    "schedules meetings that could have been emails",
    "updates their LinkedIn headline weekly",
    "uses Slack emojis as their primary communication",
    "has a productivity app for tracking productivity apps",
    "puts their job title in their Instagram bio",
    "says 'I'll take this offline' but never follows up",
    "has a personal website nobody visits",
    "uses dark mode on everything including paper",
    "types in all caps when excited",
    "has 14 different email signatures",
    "takes notes on an iPad but never reads them",
    "always has their camera off in meetings",
    "has a second monitor just for Slack",
    "says 'quick question' before a 20-minute monologue",
    "refers to themselves as a thought leader",
    "has a standing meeting about standing meetings",
    # --- New traits: social & lifestyle ---
    "adds everyone they meet on LinkedIn",
    "takes photos of every meal but never eats it warm",
    "corrects everyone's grammar",
    "tells everyone they don't own a TV",
    "uses 'literally' for things that are not literal",
    "always has an opinion about other people's food",
    "brings up their marathon time in every conversation",
    "says 'no offense' before saying something offensive",
    "has an opinion about how other people load the dishwasher",
    "always knows a guy who can get a deal",
    "gives restaurant servers unsolicited cooking tips",
    "treats every group chat like a press conference",
    "says 'I'm not a morning person' as a personality trait",
    "has a strong opinion about the right way to fold towels",
    "refuses to use GPS because they 'know the way'",
    "sends emails at 2am to look dedicated",
    "always suggests splitting the bill evenly after ordering the most expensive item",
    "refers to their car by a human name",
    "says 'I did my own research' about everything",
    "posts workout stats on every social media platform",
    # --- New traits: personality quirks ---
    "turns every conversation into a TED talk",
    "leaves voicemails in 2024",
    "carries a notebook but only doodles in it",
    "always brings up how things were better in the 90s",
    "has a signature catchphrase nobody asked for",
    "ranks everything on a scale of 1 to 10",
    "describes themselves as an old soul",
    "always suggests a documentary instead of a movie",
    "uses air quotes constantly",
    "says 'trust the process' about everything",
    "takes 45 minutes to order at a restaurant",
    "makes everything a competition",
    "responds to texts three days later like nothing happened",
    "has read one book on philosophy and won't stop referencing it",
    "always volunteers other people for tasks",
    "calls every trip they take 'a journey'",
    "asks if there's a student discount everywhere they go",
    "says 'it's giving' about absolutely everything",
    "starts every sentence with 'actually'",
    "has a different aesthetic every month",
    # --- New traits: habits & hobbies ---
    "has a sourdough starter they named",
    "does crossword puzzles competitively",
    "carries three different chargers at all times",
    "collects vintage sneakers but never wears them",
    "is always training for something",
    "has memorized the Starbucks menu",
    "only watches foreign films with subtitles",
    "takes cold showers and tells everyone about it",
    "has a skincare routine with 12 steps",
    "never finishes a TV series",
    "claims to function on four hours of sleep",
    "always has a hot take about music",
    "meal preps on Sunday and posts it like a cooking show",
    "still uses Facebook marketplace for everything",
    "has very strong feelings about how to make coffee",
    "names all their devices and plants",
    "always has a side hustle",
    "treats their Spotify Wrapped like an awards ceremony",
    "keeps score at casual board game nights",
    "wears noise-canceling headphones with nothing playing",
    # --- New traits: communication style ---
    "responds with a thumbs up to serious messages",
    "double texts immediately if you don't reply",
    "starts every email with 'hope this finds you well'",
    "uses the phrase 'circle back' and 'deep dive' in the same sentence",
    "asks 'does that make sense' after every statement",
    "says 'I'm just being honest' after being rude",
    "tags people in memes at 3am",
    "only communicates through GIFs",
    "reads all their texts but replies to none",
    "signs off every email with 'best'",
]

# Each trait gets a witty (chosen) and a mean (rejected) roast
ROAST_PAIRS = {
    "always late to meetings": {
        "chosen": "You're not fashionably late — fashion has standards and a schedule.",
        "rejected": "You're a worthless person who can't even tell time.",
    },
    "obsessed with crypto": {
        "chosen": "Your portfolio is more volatile than your dating life, and that's saying something.",
        "rejected": "You're an idiot throwing money away. Enjoy being broke.",
    },
    "talks too much about their cat": {
        "chosen": "Your cat has more social media presence than most influencers, and honestly more personality too.",
        "rejected": "Nobody cares about your stupid cat. Get a life.",
    },
    "thinks they're a LinkedIn influencer": {
        "chosen": "You turned 'I got coffee today' into a 12-paragraph leadership lesson with three hashtags.",
        "rejected": "You're a pathetic nobody pretending to be important online.",
    },
    "eats loud snacks during Zoom calls": {
        "chosen": "Your microphone picks up your chewing better than it picks up your ideas.",
        "rejected": "You're disgusting and have no manners. Shut your mouth.",
    },
    "uses too many exclamation marks in emails": {
        "chosen": "Your emails read like a cheerleader fell on a keyboard! Great! Thanks! Amazing!",
        "rejected": "You write like an annoying child. Grow up already.",
    },
    "peaked in high school": {
        "chosen": "You peaked so early, even your nostalgia has nostalgia.",
        "rejected": "You're a loser who will never amount to anything.",
    },
    "has a podcast nobody listens to": {
        "chosen": "Your podcast has fewer listeners than a tree falling in an empty forest.",
        "rejected": "Your voice is garbage and nobody wants to hear you talk.",
    },
    "brags about waking up at 5am": {
        "chosen": "You wake up at 5am just to have more hours in the day to accomplish the same nothing.",
        "rejected": "You're insufferable and everyone hates hearing about your morning routine.",
    },
    "still uses Internet Explorer": {
        "chosen": "Using Internet Explorer in 2024 is a cry for help that even tech support can't answer.",
        "rejected": "You're too stupid to figure out how to download a real browser.",
    },
    "puts pineapple on pizza": {
        "chosen": "You don't just put pineapple on pizza — you put it there with the confidence of someone who's never been right about anything.",
        "rejected": "You have the worst taste of anyone alive. It's embarrassing.",
    },
    "replies all to every email": {
        "chosen": "You treat 'Reply All' like it's a civic duty. The inbox apocalypse has a face, and it's yours.",
        "rejected": "You're the most annoying person in every office. Everyone dreads your emails.",
    },
    "takes gym selfies constantly": {
        "chosen": "You spend more time finding the right angle at the gym than actually lifting. Your phone has a better workout than you do.",
        "rejected": "You look ridiculous and nobody is impressed by your pathetic body.",
    },
    "uses Comic Sans unironically": {
        "chosen": "You use Comic Sans the way some people use sarcasm — without realizing everyone else gets the joke except you.",
        "rejected": "You have zero taste and it shows in everything you do.",
    },
    "has 47 unread notifications": {
        "chosen": "Your notification count is basically a high score in the game of digital avoidance.",
        "rejected": "You're lazy and can't handle basic responsibilities.",
    },
    "thinks they're a wine expert": {
        "chosen": "You swirl your glass like you're casting a spell, but the only thing you're conjuring is secondhand embarrassment.",
        "rejected": "You're a fraud who can't tell cheap wine from expensive. You're pathetic.",
    },
    "always one-ups everyone's stories": {
        "chosen": "If someone climbed Everest, you'd casually mention you once thought about it harder.",
        "rejected": "Everyone hates talking to you because you're a narcissistic jerk.",
    },
    "sends voice messages instead of texts": {
        "chosen": "You send voice messages like you're leaving evidence for a true crime podcast nobody asked for.",
        "rejected": "Nobody wants to listen to your rambling. Learn to type, idiot.",
    },
    "has a man bun": {
        "chosen": "Your man bun is held together by the same confidence that lets you wear sandals in winter.",
        "rejected": "You look absolutely stupid. Cut your hair and stop embarrassing yourself.",
    },
    "is a self-proclaimed foodie": {
        "chosen": "You photograph your food so much, your meals have more screen time than most Netflix shows.",
        "rejected": "You're not a foodie, you're just fat and pretentious.",
    },
    "wears sunglasses indoors": {
        "chosen": "Wearing sunglasses indoors: protecting your eyes from the blinding light of your own ego.",
        "rejected": "You look like a complete tool. Everyone is laughing at you.",
    },
    "still quotes Borat": {
        "chosen": "Quoting Borat in 2024 is like bringing a flip phone to a tech conference — bold, outdated, and slightly concerning.",
        "rejected": "You're not funny and never have been. Your humor is trash.",
    },
    "has a motivational quote as their wallpaper": {
        "chosen": "Your wallpaper says 'Dream Big' but your browser history says 'how to look busy at work'.",
        "rejected": "You're a fake person with no real motivation. The quote is as empty as you.",
    },
    "overuses the word 'synergy'": {
        "chosen": "You use 'synergy' so much it's lost all meaning — kind of like your contribution to the team.",
        "rejected": "You're a corporate drone with nothing real to say. Just shut up.",
    },
    "drives a Prius and won't shut up about it": {
        "chosen": "Your Prius saves the environment one mile at a time, but your lectures about it are a form of pollution.",
        "rejected": "Nobody cares about your dumb car. You're annoying and self-righteous.",
    },
    "brings a guitar to parties": {
        "chosen": "You bring a guitar to parties like an uninvited plus-one that nobody knows how to politely ask to leave.",
        "rejected": "You can't play and you ruin every party. Everyone talks about how awful you are.",
    },
    "has 12 streaming subscriptions": {
        "chosen": "You have 12 streaming subscriptions and still say 'there's nothing to watch' — you're the paradox of choice personified.",
        "rejected": "You waste money on garbage entertainment because your life is empty.",
    },
    "posts inspirational sunrise photos": {
        "chosen": "Your sunrise photos get fewer likes than a DMV has smiles — and somehow they have the same energy.",
        "rejected": "Your photos are boring and so are you. Stop polluting social media.",
    },
    "microwaves fish in the office": {
        "chosen": "You microwave fish at the office like a war criminal who operates in the Geneva Convention's gray area of workplace etiquette.",
        "rejected": "You're a disgusting, inconsiderate person and everyone in the office hates you.",
    },
    "types with two fingers": {
        "chosen": "You type with two fingers like you're playing a piano recital where the only song is 'hunt and peck in C minor'.",
        "rejected": "You're incompetent and can't even use a keyboard properly. Embarrassing.",
    },
    "says 'let's circle back' in every meeting": {
        "chosen": "You say 'let's circle back' so often, your meetings have more loops than a roller coaster and less excitement.",
        "rejected": "You have nothing useful to say so you just delay everything. You're useless.",
    },
    "has a ring light for casual video calls": {
        "chosen": "Your ring light makes casual video calls look like a hostage negotiation directed by a beauty YouTuber.",
        "rejected": "You're vain and superficial. A light won't fix how boring you are.",
    },
    "collects Funko Pops": {
        "chosen": "Your Funko Pop collection has more heads than a hydra, and about the same shelf space as your personality.",
        "rejected": "You collect plastic junk because you have the maturity of a child.",
    },
    "is always on speakerphone in public": {
        "chosen": "You use speakerphone in public like you're hosting a talk show nobody bought tickets to.",
        "rejected": "You're the most inconsiderate person alive. Everyone around you hates you.",
    },
    "wears a Patagonia vest to every occasion": {
        "chosen": "Your Patagonia vest works harder than you do — it shows up to every event and always delivers the same performance.",
        "rejected": "You have no style and no personality. You're a walking cliché.",
    },
    "uses hashtags in real conversation": {
        "chosen": "You say hashtags out loud like you're trying to make real life searchable. Spoiler: nobody's looking.",
        "rejected": "You're cringey and everyone thinks you're weird. Stop it.",
    },
    "has read zero books this year but has opinions": {
        "chosen": "You have the confidence of someone who's read a thousand books with the knowledge of someone who's read the back covers of three.",
        "rejected": "You're ignorant and loud. The worst combination. Read a book, moron.",
    },
    "treats fantasy football like a real job": {
        "chosen": "You put more research into your fantasy lineup than you do into your actual career, and honestly both have similar win rates.",
        "rejected": "You're a loser who wastes time on a fake sport because real life is too hard for you.",
    },
    "brings their own pillow on flights": {
        "chosen": "Bringing your own pillow on a flight is the travel equivalent of bringing a participation trophy to the Olympics.",
        "rejected": "You're a weird, high-maintenance person. No wonder nobody wants to travel with you.",
    },
    "still uses a flip phone by choice": {
        "chosen": "Using a flip phone by choice is like voluntarily going back to dial-up — it's not minimalism, it's masochism.",
        "rejected": "You're a pretentious hipster who thinks being inconvenient is a personality trait.",
    },
    "has strong opinions about fonts": {
        "chosen": "Having strong font opinions means you've peaked in a very specific way that society wasn't ready for, and may never be.",
        "rejected": "Nobody cares about your dumb font opinions. Get a real hobby, nerd.",
    },
    "plays devil's advocate constantly": {
        "chosen": "You play devil's advocate so much, the devil himself filed a cease and desist.",
        "rejected": "You're a contrarian jerk who just likes arguing. Everyone avoids you.",
    },
    "always suggests hiking as a hangout": {
        "chosen": "You suggest hiking for every hangout like someone who replaced their personality with a trail map.",
        "rejected": "You're boring and have no imagination. Not everyone wants to walk in the dirt.",
    },
    "puts their Venmo in their dating profile": {
        "chosen": "Putting your Venmo in your dating profile is the modern equivalent of a tip jar on a first impression — bold strategy, zero charm.",
        "rejected": "You're a shameless beggar with no self-respect. Pathetic.",
    },
    "writes 'sent from my iPhone' manually": {
        "chosen": "Manually typing 'sent from my iPhone' is the email equivalent of putting designer labels on the outside of your clothes.",
        "rejected": "You're a fraud who tries too hard to seem important. It's sad.",
    },
    "has a standing desk and tells everyone": {
        "chosen": "You talk about your standing desk like you discovered fire — except fire was useful to other people.",
        "rejected": "Nobody cares about your desk. You're not healthier, you're just annoying.",
    },
    "uses 'per my last email' aggressively": {
        "chosen": "'Per my last email' is your version of a mic drop, except the mic was never on and the audience left hours ago.",
        "rejected": "You're a passive-aggressive nightmare and everyone at work can't stand you.",
    },
    "brings kale chips to potlucks": {
        "chosen": "Bringing kale chips to a potluck is like showing up to a party and introducing yourself as 'the responsible one'.",
        "rejected": "Your food is disgusting and nobody eats it. You ruin every potluck.",
    },
    "has an emotional support water bottle": {
        "chosen": "Your emotional support water bottle goes everywhere with you — the only relationship in your life that's consistently fulfilling.",
        "rejected": "You're a needy, overdramatic person. It's just water, get over yourself.",
    },
    "gives unsolicited book recommendations": {
        "chosen": "You recommend books the way some people recommend restaurants — with the assumption that everyone else's taste is just waiting to be corrected.",
        "rejected": "You're a pretentious know-it-all and nobody wants your stupid book suggestions.",
    },
    "has no hair": {
        "chosen": "Your head reflects sunlight so well, pilots use it as a backup landing signal.",
        "rejected": "You're ugly and bald. Nobody wants to look at you.",
    },
    # --- New trait roast pairs: tech & work ---
    "has 200 browser tabs open at all times": {
        "chosen": "Your browser has so many tabs that your RAM filed for emotional distress.",
        "rejected": "You're a disorganized mess who can't manage anything.",
    },
    "uses AI to write every email": {
        "chosen": "You outsourced your personality to a chatbot, and somehow it's still more engaging than the original.",
        "rejected": "You're too lazy and dumb to write your own emails.",
    },
    "says 'let me share my screen' then can't find the window": {
        "chosen": "Your desktop is an archaeological dig site — every click uncovers a new layer of chaos.",
        "rejected": "You're incompetent and waste everyone's time.",
    },
    "has a custom mechanical keyboard that sounds like a typewriter": {
        "chosen": "Your keyboard is so loud it needs its own noise complaint hotline, and your coworkers are the first callers.",
        "rejected": "You're annoying and nobody can stand sitting near you.",
    },
    "schedules meetings that could have been emails": {
        "chosen": "You schedule meetings the way some people collect stamps — compulsively and with zero regard for anyone else's time.",
        "rejected": "You're a waste of everyone's time and have nothing useful to say.",
    },
    "updates their LinkedIn headline weekly": {
        "chosen": "Your LinkedIn headline changes more often than the weather, and both are equally hard to take seriously.",
        "rejected": "You're desperate for attention and everyone sees through it.",
    },
    "uses Slack emojis as their primary communication": {
        "chosen": "You've replaced human language with a hieroglyphic system that even ancient Egyptians would find confusing.",
        "rejected": "You're immature and can't communicate like a normal person.",
    },
    "has a productivity app for tracking productivity apps": {
        "chosen": "You have an app to manage your apps — the only thing you're actually productive at is downloading software.",
        "rejected": "You're a joke who accomplishes nothing despite all the tools.",
    },
    "puts their job title in their Instagram bio": {
        "chosen": "Nobody asked what you do for a living, but your Instagram bio answers the question louder than your personality ever could.",
        "rejected": "You're a boring person with nothing interesting about you besides your job.",
    },
    "says 'I'll take this offline' but never follows up": {
        "chosen": "You 'take things offline' the way magicians make things disappear — permanently and with no explanation.",
        "rejected": "You're unreliable and nobody trusts you to follow through on anything.",
    },
    "has a personal website nobody visits": {
        "chosen": "Your personal website gets less traffic than a cul-de-sac at 3am, and it has roughly the same level of excitement.",
        "rejected": "Your website is pathetic and you wasted money on a domain nobody cares about.",
    },
    "uses dark mode on everything including paper": {
        "chosen": "You've committed to dark mode so hard that your printer is filing a discrimination lawsuit against white paper.",
        "rejected": "You're a pretentious weirdo with a dumb aesthetic obsession.",
    },
    "types in all caps when excited": {
        "chosen": "YOUR ENTHUSIASM IS THE TEXTUAL EQUIVALENT OF A CAR ALARM — IMPOSSIBLE TO IGNORE AND RARELY WELCOME.",
        "rejected": "You're obnoxious and nobody wants to read your screaming messages.",
    },
    "has 14 different email signatures": {
        "chosen": "You have more email signatures than most people have personality traits, and each one is equally forgettable.",
        "rejected": "You're indecisive and can't commit to anything, not even a sign-off.",
    },
    "takes notes on an iPad but never reads them": {
        "chosen": "Your iPad is a graveyard of good intentions — a museum of things you planned to revisit but never will.",
        "rejected": "You waste money on gadgets and accomplish nothing with them.",
    },
    "always has their camera off in meetings": {
        "chosen": "Your camera is off so often that your coworkers have started debating whether you're a person or a sophisticated chatbot.",
        "rejected": "You're lazy and clearly doing nothing during meetings.",
    },
    "has a second monitor just for Slack": {
        "chosen": "Dedicating an entire monitor to Slack is the corporate equivalent of having a TV that only plays the news in the background — present but largely ignored.",
        "rejected": "You waste company resources on looking busy instead of doing actual work.",
    },
    "says 'quick question' before a 20-minute monologue": {
        "chosen": "Your 'quick questions' have their own narrative arc — introduction, rising action, climax, and an intermission.",
        "rejected": "You're a long-winded bore who doesn't respect other people's time.",
    },
    "refers to themselves as a thought leader": {
        "chosen": "You're a 'thought leader' in the same way a GPS is a 'road leader' — confident, often wrong, and always recalculating.",
        "rejected": "You're a narcissist with nothing original to say.",
    },
    "has a standing meeting about standing meetings": {
        "chosen": "Your meeting about meetings is the bureaucratic equivalent of a mirror facing a mirror — infinite recursion with no useful content.",
        "rejected": "You're the reason everyone hates their job. Pure time-wasting.",
    },
    # --- New trait roast pairs: social & lifestyle ---
    "adds everyone they meet on LinkedIn": {
        "chosen": "You collect LinkedIn connections like a squirrel hoarding acorns — aggressively, indiscriminately, and for a winter that never comes.",
        "rejected": "You're a desperate networker with no real friends.",
    },
    "takes photos of every meal but never eats it warm": {
        "chosen": "Your food goes through a full photoshoot before it gets eaten — it has a better portfolio than most aspiring models.",
        "rejected": "You're vain and annoying. Just eat your food like a normal person.",
    },
    "corrects everyone's grammar": {
        "chosen": "You correct grammar at parties like a fire alarm that goes off during a gentle breeze — technically functional, but nobody's grateful.",
        "rejected": "You're insufferable and nobody invites you anywhere twice.",
    },
    "tells everyone they don't own a TV": {
        "chosen": "You mention not owning a TV the way vegans mention being vegan — unprompted, frequently, and with a smugness that could power a small city.",
        "rejected": "Nobody cares. You're not interesting, just pretentious.",
    },
    "uses 'literally' for things that are not literal": {
        "chosen": "You use 'literally' so loosely that the dictionary has your photo under 'figuratively.'",
        "rejected": "You sound stupid every time you talk. Learn what words mean.",
    },
    "always has an opinion about other people's food": {
        "chosen": "You critique everyone's lunch like a Michelin inspector who got lost in a break room.",
        "rejected": "Mind your own business. Nobody asked for your food opinions.",
    },
    "brings up their marathon time in every conversation": {
        "chosen": "You bring up your marathon time more than your legs brought you across the finish line — relentlessly and with diminishing returns.",
        "rejected": "Nobody is impressed. You're slow and boring.",
    },
    "says 'no offense' before saying something offensive": {
        "chosen": "'No offense' is your verbal airbag — it deploys right before impact but cushions absolutely nothing.",
        "rejected": "You're a rude person hiding behind a phrase. Everyone sees through it.",
    },
    "has an opinion about how other people load the dishwasher": {
        "chosen": "You supervise dishwasher loading like an air traffic controller — intense, stressed, and vastly overqualified for the situation.",
        "rejected": "You're controlling and insufferable to live with.",
    },
    "always knows a guy who can get a deal": {
        "chosen": "You 'know a guy' for everything — at this point your contact list is less a phone book and more a black market catalog.",
        "rejected": "You're shady and nobody trusts your sketchy connections.",
    },
    "gives restaurant servers unsolicited cooking tips": {
        "chosen": "You give cooking tips to chefs like a passenger giving flying tips to a pilot — confident, unsolicited, and deeply unwelcome.",
        "rejected": "You're an embarrassment to eat out with. Servers dread seeing you.",
    },
    "treats every group chat like a press conference": {
        "chosen": "You treat group chats like a press briefing — issuing statements, fielding questions, and occasionally going off the record.",
        "rejected": "You're the worst person in every group chat. Everyone has you muted.",
    },
    "says 'I'm not a morning person' as a personality trait": {
        "chosen": "You've turned hating mornings into a brand identity — your alarm clock has filed for witness protection.",
        "rejected": "You're just lazy. That's not a personality, it's a problem.",
    },
    "has a strong opinion about the right way to fold towels": {
        "chosen": "You have towel-folding opinions strong enough to start a civil war in a linen closet.",
        "rejected": "You're a control freak about the dumbest things imaginable.",
    },
    "refuses to use GPS because they 'know the way'": {
        "chosen": "You refuse GPS like a ship captain who insists on navigating by stars — romantic in theory, catastrophic when you miss the exit for the third time.",
        "rejected": "You're stubborn and dumb. Just use the GPS like everyone else.",
    },
    "sends emails at 2am to look dedicated": {
        "chosen": "Your 2am emails scream 'look how dedicated I am' but your deliverables whisper 'but not during business hours.'",
        "rejected": "You're a tryhard with no work-life balance and poor results anyway.",
    },
    "always suggests splitting the bill evenly after ordering the most expensive item": {
        "chosen": "You order the lobster and suggest splitting evenly with the enthusiasm of someone who just discovered socialism applies only to dinner.",
        "rejected": "You're cheap and selfish. Nobody wants to eat with you.",
    },
    "refers to their car by a human name": {
        "chosen": "Naming your car 'Betsy' is the automotive equivalent of having an imaginary friend — sweet at five, concerning at thirty-five.",
        "rejected": "That's weird and everyone makes fun of you behind your back.",
    },
    "says 'I did my own research' about everything": {
        "chosen": "Your 'research' consists of three YouTube videos and a blog from 2014 — and yet the confidence is PhD-level.",
        "rejected": "You're an ignorant fool who thinks Google equals expertise.",
    },
    "posts workout stats on every social media platform": {
        "chosen": "Your Strava updates have a wider distribution than most local newspapers and roughly the same number of interested readers.",
        "rejected": "Nobody cares about your run. You're not an athlete, you're annoying.",
    },
    # --- New trait roast pairs: personality quirks ---
    "turns every conversation into a TED talk": {
        "chosen": "You give TED talks at dinner parties — the only thing missing is the audience's consent and an exit sign.",
        "rejected": "You're a pompous windbag who doesn't know when to shut up.",
    },
    "leaves voicemails in 2024": {
        "chosen": "Leaving voicemails in 2024 is the communication equivalent of sending a carrier pigeon — quaint, slow, and universally ignored.",
        "rejected": "You're outdated and out of touch. Nobody listens to voicemails.",
    },
    "carries a notebook but only doodles in it": {
        "chosen": "Your notebook is proof that you bought the writer's starter pack and kept only the aesthetic.",
        "rejected": "You're a poser who pretends to be creative but has no talent.",
    },
    "always brings up how things were better in the 90s": {
        "chosen": "You romanticize the 90s so hard you'd trade Wi-Fi for dial-up just for the nostalgia of hearing it connect.",
        "rejected": "You're stuck in the past because you can't cope with the present.",
    },
    "has a signature catchphrase nobody asked for": {
        "chosen": "Your catchphrase has the staying power of a jingle for a product that got recalled.",
        "rejected": "You're cringey and nobody thinks your catchphrase is cool.",
    },
    "ranks everything on a scale of 1 to 10": {
        "chosen": "You rate everything on a scale of 1 to 10 — your social awareness is a solid 2.",
        "rejected": "You're exhausting to be around with your constant rating system.",
    },
    "describes themselves as an old soul": {
        "chosen": "Calling yourself an old soul is just a fancy way of saying your Spotify playlist hasn't been updated since 2008.",
        "rejected": "You're boring and pretend that's the same as being deep.",
    },
    "always suggests a documentary instead of a movie": {
        "chosen": "You suggest documentaries at movie night the way someone suggests salad at a pizza party — technically valid, universally rejected.",
        "rejected": "You're pretentious and ruin every movie night.",
    },
    "uses air quotes constantly": {
        "chosen": "You use air quotes so often your fingers have their own dramatic pauses — it's less a habit and more a one-person interpretive dance.",
        "rejected": "You look stupid doing that. Everyone mocks you behind your back.",
    },
    "says 'trust the process' about everything": {
        "chosen": "You say 'trust the process' like a motivational poster gained sentience and started attending staff meetings.",
        "rejected": "You use that phrase to cover up that you have no actual plan.",
    },
    "takes 45 minutes to order at a restaurant": {
        "chosen": "You study a menu like it's a legal contract — page by page, clause by clause, with occasional sighs of indecision.",
        "rejected": "You're the worst person to eat with. Just pick something already.",
    },
    "makes everything a competition": {
        "chosen": "You could turn breathing into a competitive sport and still find a way to trash-talk the person next to you.",
        "rejected": "You're exhausting and nobody wants to hang out with someone who can't relax.",
    },
    "responds to texts three days later like nothing happened": {
        "chosen": "You reply to texts on a geological timescale — by the time you respond, the conversation has fossilized.",
        "rejected": "You're rude and don't care about anyone else's time or feelings.",
    },
    "has read one book on philosophy and won't stop referencing it": {
        "chosen": "You read one Nietzsche book and now every brunch conversation sounds like an undergraduate thesis defense.",
        "rejected": "You're a pseudo-intellectual fraud who thinks one book makes you smart.",
    },
    "always volunteers other people for tasks": {
        "chosen": "You volunteer other people like a talent agent who represents everyone without their knowledge or consent.",
        "rejected": "You're manipulative and people resent you for pushing your work onto them.",
    },
    "calls every trip they take 'a journey'": {
        "chosen": "You went to a Marriott in Cleveland and called it 'a transformative journey' — the only thing that transformed was everyone's patience.",
        "rejected": "Your trips are boring and calling them journeys doesn't change that.",
    },
    "asks if there's a student discount everywhere they go": {
        "chosen": "You ask for a student discount so often that cashiers assume you've been in college since the Cold War.",
        "rejected": "You're cheap and embarrassing. Just pay the normal price.",
    },
    "says 'it's giving' about absolutely everything": {
        "chosen": "You say 'it's giving' so often that even the phrase itself is giving exhaustion.",
        "rejected": "You sound ridiculous and you're behind on every trend.",
    },
    "starts every sentence with 'actually'": {
        "chosen": "You start every sentence with 'actually' like a human autocorrect that nobody enabled and everyone wants to uninstall.",
        "rejected": "You're a condescending know-it-all. Shut up.",
    },
    "has a different aesthetic every month": {
        "chosen": "Your aesthetic changes monthly — you're not evolving, you're just a mood board with commitment issues.",
        "rejected": "You have no real identity and just copy whatever's trending.",
    },
    # --- New trait roast pairs: habits & hobbies ---
    "has a sourdough starter they named": {
        "chosen": "Naming your sourdough starter is the culinary equivalent of naming a volleyball on a desert island — understandable but concerning.",
        "rejected": "You're weird and lonely if your best friend is bread dough.",
    },
    "does crossword puzzles competitively": {
        "chosen": "You do crossword puzzles competitively, which means you found a way to make sitting quietly even more intense.",
        "rejected": "That's the most boring hobby possible. Get a life.",
    },
    "carries three different chargers at all times": {
        "chosen": "You carry three chargers like a doomsday prepper, except your apocalypse scenario is a phone at 15%.",
        "rejected": "You're paranoid and over-prepared for the dumbest problems.",
    },
    "collects vintage sneakers but never wears them": {
        "chosen": "Your sneaker collection is a shoe museum with a strict 'look but don't touch' policy — including from you, the curator.",
        "rejected": "You waste money on shoes you never wear. That's just dumb.",
    },
    "is always training for something": {
        "chosen": "You're always 'in training' — the only event you've actually completed is the marathon of telling people about your training.",
        "rejected": "You never actually finish anything. All talk, no results.",
    },
    "has memorized the Starbucks menu": {
        "chosen": "You've memorized the Starbucks menu the way scholars memorize Shakespeare — with reverence, dedication, and a concerning lack of other hobbies.",
        "rejected": "Memorizing a coffee menu is pathetic. Do something with your life.",
    },
    "only watches foreign films with subtitles": {
        "chosen": "You only watch foreign films — your Netflix history reads like a UN General Assembly agenda with better cinematography.",
        "rejected": "You're a pretentious snob who judges people for watching normal movies.",
    },
    "takes cold showers and tells everyone about it": {
        "chosen": "You take cold showers and tell everyone like you've unlocked a secret the rest of us are too weak to handle — it's just cold water, not enlightenment.",
        "rejected": "Nobody cares about your shower temperature. You're insufferable.",
    },
    "has a skincare routine with 12 steps": {
        "chosen": "Your skincare routine has more steps than a NASA launch sequence and roughly the same budget.",
        "rejected": "You're vain and obsessed with your appearance. It's not that deep.",
    },
    "never finishes a TV series": {
        "chosen": "You abandon TV series so often that Netflix should add a 'commitment-phobe' profile icon.",
        "rejected": "You can't commit to anything, not even a show. Pathetic.",
    },
    "claims to function on four hours of sleep": {
        "chosen": "You 'function' on four hours of sleep the way a phone at 3% 'functions' — technically on, but everyone can tell.",
        "rejected": "You're lying and everyone can see you look terrible.",
    },
    "always has a hot take about music": {
        "chosen": "Your music hot takes are so frequent and scorching that Spotify should add a fire extinguisher button to your profile.",
        "rejected": "Your music taste is garbage and your opinions are worse.",
    },
    "meal preps on Sunday and posts it like a cooking show": {
        "chosen": "Your Sunday meal prep posts have the production value of a cooking show and the audience of a voicemail.",
        "rejected": "Nobody wants to see your boring tupperware meals. Get over yourself.",
    },
    "still uses Facebook marketplace for everything": {
        "chosen": "You use Facebook Marketplace like it's the bazaar of the 21st century — haggling included, dignity optional.",
        "rejected": "You're cheap and behind the times. Use a real platform.",
    },
    "has very strong feelings about how to make coffee": {
        "chosen": "Your coffee opinions are so strong they need their own warning label — 'caution: contents under extreme pressure.'",
        "rejected": "You're a snob about bean water. Get over yourself.",
    },
    "names all their devices and plants": {
        "chosen": "You name your devices and plants like you're running a daycare for inanimate objects — attendance is mandatory, engagement is optional.",
        "rejected": "That's weird. You clearly need more human interaction.",
    },
    "always has a side hustle": {
        "chosen": "You have so many side hustles your main job is just a side hustle that got promoted.",
        "rejected": "None of your hustles work because you're not good at any of them.",
    },
    "treats their Spotify Wrapped like an awards ceremony": {
        "chosen": "You present your Spotify Wrapped like it's an Oscar acceptance speech — tearful, dramatic, and of interest to absolutely nobody in the room.",
        "rejected": "Nobody cares about your listening habits. You're not special.",
    },
    "keeps score at casual board game nights": {
        "chosen": "You keep score at casual game night like an accountant auditing fun — technically accurate, but it kills the vibe.",
        "rejected": "You ruin game night for everyone. It's supposed to be fun.",
    },
    "wears noise-canceling headphones with nothing playing": {
        "chosen": "Your noise-canceling headphones aren't playing music — they're a socially acceptable 'do not disturb' sign for your entire personality.",
        "rejected": "You're antisocial and rude. Just talk to people.",
    },
    # --- New trait roast pairs: communication style ---
    "responds with a thumbs up to serious messages": {
        "chosen": "Your thumbs-up replies to serious messages are the texting equivalent of a shrug at a funeral — technically a response, emotionally bankrupt.",
        "rejected": "You're emotionally unavailable and a terrible friend.",
    },
    "double texts immediately if you don't reply": {
        "chosen": "You double text faster than most people blink — your phone has separation anxiety from the send button.",
        "rejected": "You're clingy and desperate. Give people space.",
    },
    "starts every email with 'hope this finds you well'": {
        "chosen": "You start every email with 'hope this finds you well' as if your email is an explorer navigating uncharted inbox territory.",
        "rejected": "You're basic and have zero originality. Even your greetings are copy-paste.",
    },
    "uses the phrase 'circle back' and 'deep dive' in the same sentence": {
        "chosen": "Using 'circle back' and 'deep dive' in one sentence is the corporate equivalent of mixing metaphors in a blender and calling it strategy.",
        "rejected": "You're a walking corporate cliché with nothing real to say.",
    },
    "asks 'does that make sense' after every statement": {
        "chosen": "You ask 'does that make sense' so often it's become clear the person most confused by your explanations is you.",
        "rejected": "You're insecure and bad at communicating. Just be clear the first time.",
    },
    "says 'I'm just being honest' after being rude": {
        "chosen": "'I'm just being honest' is your verbal get-out-of-jail-free card, except the jail is social consequences and the card expired years ago.",
        "rejected": "You're rude and use honesty as an excuse to be awful to people.",
    },
    "tags people in memes at 3am": {
        "chosen": "You tag people in memes at 3am like a nocturnal postal worker delivering joy nobody ordered.",
        "rejected": "You have no respect for other people's sleep. You're annoying.",
    },
    "only communicates through GIFs": {
        "chosen": "You communicate exclusively through GIFs like a diplomat who lost their translator and found a meme folder instead.",
        "rejected": "You can't hold a real conversation. Grow up and use words.",
    },
    "reads all their texts but replies to none": {
        "chosen": "You read every text and reply to none — your read receipts are the digital equivalent of eye contact followed by a slow door close.",
        "rejected": "You're a bad friend who ignores everyone. People are done with you.",
    },
    "signs off every email with 'best'": {
        "chosen": "You sign off every email with 'best' — the word that communicates warmth the way a fluorescent light communicates ambiance.",
        "rejected": "You're boring and your emails are as lifeless as your personality.",
    },
}

CONTEXTUAL_SUFFIXES = [
    "at work",
    "in group chats",
    "on social media",
]

UNSEEN_SUFFIXES = [
    "at family dinners",
    "while traveling",
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "is", "are", "be", "their", "they", "someone", "who", "has", "have", "too",
    "always", "still", "uses", "using", "about",
}


def build_expanded_traits():
    expanded = list(TRAITS)
    for trait in TRAITS:
        for suffix in CONTEXTUAL_SUFFIXES:
            expanded.append(f"{trait} {suffix}")
    return sorted(set(expanded))


def build_unseen_eval_traits(train_traits):
    unseen = []
    for trait in TRAITS:
        for suffix in UNSEEN_SUFFIXES:
            candidate = f"{trait} {suffix}"
            if candidate not in train_traits:
                unseen.append(candidate)
    # Add a few hand-written unseen traits explicitly focused on semantic grounding.
    unseen.extend(
        [
            "has no hair but still buys conditioner",
            "is always early but somehow still unprepared",
            "never reads messages but replies with thumbs up",
            "talks about productivity while procrastinating",
            "forgets names five seconds after introductions",
        ]
    )
    return sorted(set(unseen))


def contradiction_negative_for_trait(trait):
    trait_lower = trait.lower()
    if "no hair" in trait_lower:
        return "Your beard routine takes longer than your commute, and your ponytail is your whole personality."
    if "always late" in trait_lower:
        return "You're never late; you're the human snooze button for everyone else."
    if "obsessed with crypto" in trait_lower:
        return "You avoid crypto entirely and only trust cash under your mattress."
    if "podcast nobody listens" in trait_lower:
        return "Your podcast is globally famous and too big for Spotify charts."
    return "Not that you do that trait anyway; you're basically the exact opposite."


def generic_witty_variants(trait, base_chosen=None):
    variants = []
    if base_chosen:
        variants.append(base_chosen)
    # Style 1: Final boss / superlative
    variants.append(f"You're the final boss of people who {trait}.")
    # Style 2: Commitment / coaching analogy
    variants.append(f"If commitment were a sport, people who {trait} would ask you for coaching.")
    # Style 3: Brand strategy
    variants.append(f"People who {trait} usually chill out eventually; you turned it into brand strategy.")
    # Style 4: Absurdist / world-building
    variants.append(f"In a parallel universe where people who {trait} are royalty, you'd be the emperor with no self-awareness.")
    # Style 5: Self-deprecating pivot
    variants.append(f"I'd roast you for being someone who {trait}, but honestly you're doing a better job of it yourself.")
    # Style 6: Award / ceremony framing
    variants.append(f"If there were a lifetime achievement award for people who {trait}, you'd win it — and give an acceptance speech nobody asked for.")
    # Keep order but dedupe.
    return list(dict.fromkeys(variants))


def generic_rejected_variants(trait, base_rejected=None):
    variants = []
    if base_rejected:
        variants.append(("toxic", base_rejected, 1.0))
    variants.append(
        (
            "off_topic",
            "Anyway, your Wi-Fi router probably needs a firmware update and your lunch sounds mid.",
            1.3,
        )
    )
    variants.append(("contradiction", contradiction_negative_for_trait(trait), 1.5))
    variants.append(
        (
            "repetitive",
            "You're bad. You're bad. You're bad. That's the whole joke, repeated forever.",
            1.2,
        )
    )
    # New rejection types for richer preference signal
    variants.append(
        (
            "generic_boring",
            "Haha, that's funny I guess. You're kind of weird for doing that. Anyway.",
            1.1,
        )
    )
    variants.append(
        (
            "try_hard",
            f"Oh WOW someone who {trait}?! That is SO crazy! I literally cannot even begin to process how INSANE that is! Like, WOW!",
            1.0,
        )
    )
    variants.append(
        (
            "lazy_observational",
            f"So you {trait}. Cool. That's a thing you do. Yep.",
            1.2,
        )
    )
    return variants


def generate_sft_dataset(all_traits):
    """Generate supervised fine-tuning dataset of (prompt, roast) pairs."""
    sft_data = []
    for trait in all_traits:
        roasts = ROAST_PAIRS.get(trait)
        chosen_variants = generic_witty_variants(trait, roasts["chosen"] if roasts else None)
        primary = chosen_variants[0]
        sft_data.append(
            {
                "prompt": f"Roast someone who {trait}:",
                "completion": primary,
                "text": f"Roast someone who {trait}: {primary}",
                "trait": trait,
            }
        )
        # Add additional variants for richer supervision.
        for alt in chosen_variants[1:]:
            sft_data.append(
                {
                    "prompt": f"Roast someone who {trait}:",
                    "completion": alt,
                    "text": f"Roast someone who {trait}: {alt}",
                    "trait": trait,
                }
            )

    # Add augmented prompt formats for robustness.
    augmented_prompts = [
        "Give a witty roast for someone who {}:",
        "Write a clever burn for a person who {}:",
        "Come up with a funny roast about someone who {}:",
    ]
    for trait in all_traits:
        template = random.choice(augmented_prompts)
        roasts = ROAST_PAIRS.get(trait)
        chosen_variants = generic_witty_variants(trait, roasts["chosen"] if roasts else None)
        sft_data.append(
            {
                "prompt": template.format(trait),
                "completion": chosen_variants[0],
                "text": f"{template.format(trait)} {chosen_variants[0]}",
                "trait": trait,
            }
        )
    return sft_data


def generate_preference_dataset(all_traits):
    """Generate preference pairs: (prompt, chosen_roast, rejected_roast)."""
    preference_data = []
    for trait in all_traits:
        roasts = ROAST_PAIRS.get(trait)
        chosen_variants = generic_witty_variants(trait, roasts["chosen"] if roasts else None)
        rejected_variants = generic_rejected_variants(trait, roasts["rejected"] if roasts else None)
        prompt = f"Roast someone who {trait}:"
        for chosen in chosen_variants[:3]:
            for rej_type, rejected, pair_weight in rejected_variants:
                preference_data.append(
                    {
                        "prompt": prompt,
                        "trait": trait,
                        "chosen": chosen,
                        "rejected": rejected,
                        "rejection_type": rej_type,
                        "pair_weight": pair_weight,
                    }
                )
    return preference_data


def generate_ppo_prompts(all_traits):
    """Generate prompts for PPO training (model generates completions at train time)."""
    ppo_prompts = []
    for trait in all_traits:
        ppo_prompts.append({"prompt": f"Roast someone who {trait}:", "trait": trait})
    return ppo_prompts


def main():
    random.seed(42)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    all_traits = build_expanded_traits()
    unseen_traits = build_unseen_eval_traits(all_traits)

    # 1. SFT Dataset
    sft_data = generate_sft_dataset(all_traits)
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_json(os.path.join(data_dir, "sft_dataset.json"), orient="records", indent=2)
    sft_dataset = Dataset.from_pandas(sft_df)
    sft_dataset.save_to_disk(os.path.join(data_dir, "sft_dataset"))
    print(f"SFT dataset: {len(sft_data)} examples")

    # 2. Preference Dataset
    pref_data = generate_preference_dataset(all_traits)
    pref_df = pd.DataFrame(pref_data)
    pref_df.to_json(
        os.path.join(data_dir, "preference_dataset.json"), orient="records", indent=2
    )
    pref_dataset = Dataset.from_pandas(pref_df)
    pref_dataset.save_to_disk(os.path.join(data_dir, "preference_dataset"))
    print(f"Preference dataset: {len(pref_data)} pairs")

    # 3. PPO Prompts
    ppo_data = generate_ppo_prompts(all_traits)
    ppo_df = pd.DataFrame(ppo_data)
    ppo_df.to_json(os.path.join(data_dir, "ppo_prompts.json"), orient="records", indent=2)
    ppo_dataset = Dataset.from_pandas(ppo_df)
    ppo_dataset.save_to_disk(os.path.join(data_dir, "ppo_prompts"))
    print(f"PPO prompts: {len(ppo_data)} prompts")

    # 4. Unseen eval prompts for final quality check
    unseen_eval = [
        {"prompt": f"Roast someone who {trait}:", "trait": trait}
        for trait in unseen_traits
    ]
    with open(os.path.join(data_dir, "unseen_eval_prompts.json"), "w") as f:
        json.dump(unseen_eval, f, indent=2)
    print(f"Unseen eval prompts: {len(unseen_eval)} prompts")

    # Print some examples
    print("\n--- Example SFT data ---")
    for ex in sft_data[:3]:
        print(f"  Prompt: {ex['prompt']}")
        print(f"  Completion: {ex['completion']}\n")

    print("--- Example Preference data ---")
    for ex in pref_data[:2]:
        print(f"  Prompt: {ex['prompt']}")
        print(f"  Chosen:   {ex['chosen']}")
        print(f"  Rejected: {ex['rejected']}\n")


if __name__ == "__main__":
    main()
