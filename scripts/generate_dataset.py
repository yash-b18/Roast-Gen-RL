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
    variants.extend(
        [
            f"You're the final boss of people who {trait}.",
            f"If commitment were a sport, people who {trait} would ask you for coaching.",
            f"People who {trait} usually chill out eventually; you turned it into brand strategy.",
        ]
    )
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
