#!/usr/bin/env python3
"""One self-contained Python cell for the historical tidbits project.
"""

# Imports from the Python standard library.
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

# Settings for this single cell. Installation uses this Python interpreter.
INSTALL_MISSING_DEPENDENCIES = True
REQUIREMENTS = [
    "packaging>=23",
    "torch>=2.6,<3",
    "transformers==5.2.0",
    "tokenizers>=0.22,<0.23",
    "numpy>=1.26,<3",
    "requests>=2.32.5,<3",
]
CONFIG = {
    "prepare": True,
    "train": True,
    "evaluate": True,
    "interactive": True,
    "data_dir": "data/one_piece/prepared",
    "cache_dir": "data/one_piece/raw",
    "output_dir": "runs/one_piece",
    "model_dir": None,  # None selects output_dir/best, then output_dir/model.
    "resume": None,  # None automatically finds a checkpoint in output_dir.
    "mirror": "https://mirror.cs.odu.edu/gutenberg-epub",
    "download_delay": 1.0,
    "max_sources": 0,  # 0 means all embedded sources; a pilot needs >= 3.
    "skip_errors": False,
    "vocab_size": 8192,
    "validation_fraction": 0.05,
    "test_fraction": 0.05,
    "seed": 42,
    "device": "auto",  # auto, cuda, mps, or cpu.
    "threads": 4,
    "max_steps": 10000,
    "batch_size": 8,
    "gradient_accumulation": 4,
    "context": 256,
    "layers": 6,
    "width": 320,
    "heads": 5,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 200,
    "eval_every": 500,
    "eval_batches": 20,  # Training validation; 0 evaluates the full split.
    "save_every": 1000,
    "log_every": 25,
    "sample_max_new_tokens": 64,
    "sample_prompts": [
        "An eclipse occurs when",
        "The purpose of a thermometer is",
        "Q. What is steam?\nA.",
        "In gardening, it is important to",
    ],
    "split": "test",
    "max_batches": 20,  # Final evaluation; 0 evaluates the full held-out split.
    "max_new_tokens": 64,  # Interactive and evaluation response length.
    "temperature": 0.8,
    "report": None,  # None writes a test-results.json beside the chosen model.
}


def ensure_dependencies(install=True):
    """Install missing/incompatible packages without reinstalling a working stack."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        needed = list(REQUIREMENTS)
    else:
        needed = []
        for specification in REQUIREMENTS:
            requirement = Requirement(specification)
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                needed.append(specification)
            else:
                if version not in requirement.specifier:
                    needed.append(specification)
    if not needed:
        return
    command = [sys.executable, "-m", "pip", "install", *needed]
    if not install:
        raise RuntimeError(
            "Install the required dependencies first: " + " ".join(command)
        )
    loaded = [
        specification
        for specification in needed
        if re.split(r"[<>=!~]", specification, maxsplit=1)[0] in sys.modules
    ]
    print("Installing dependencies: " + ", ".join(needed), flush=True)
    subprocess.check_call(command)
    importlib.invalidate_caches()
    if loaded:
        raise RuntimeError(
            "Updated packages that were already imported. Restart the notebook kernel "
            "and rerun this cell so the newly installed versions are used."
        )


if __name__ == "__main__":
    ensure_dependencies(INSTALL_MISSING_DEPENDENCIES)

# Third-party imports are intentionally after the optional dependency bootstrap.
import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from urllib3.util.retry import Retry

"""For (hopefully) historical tidbits of information"""
sources = [
  "https://www.gutenberg.org/cache/epub/57493/pg57493.txt", #The Natural History of Pliny, Volume 1 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/60230/pg60230.txt", #The Natural History of Pliny, Volume 2 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/59131/pg59131.txt", #The Natural History of Pliny, Volume 3 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/61113/pg61113.txt", #The Natural History of Pliny, Volume 4 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/60688/pg60688.txt", #The Natural History of Pliny, Volume 5 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/62704/pg62704.txt", #The Natural History of Pliny, Volume 6 (of 6) by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/58589/pg58589.txt", #The Boys' and Girls' Pliny by the Elder Pliny
  "https://www.gutenberg.org/cache/epub/26000/pg26000.txt", #The Sailor's Word-Book by W. H. Smyth
  "https://www.gutenberg.org/cache/epub/42474/pg42474.txt", #1000 Mythological Characters Briefly Described by Edward Sylvester Ellis
  "https://www.gutenberg.org/cache/epub/55059/pg55059.txt", #Modern English biography, volume 1 (of 6), A-H by Frederic Boase
  "https://www.gutenberg.org/cache/epub/71046/pg71046.txt", #Modern English biography, volume 2 (of 6), I-Q by Frederic Boase
  "https://www.gutenberg.org/cache/epub/20776/pg20776.txt", #Encyclopedia of Needlework by Thérèse de Dillmont
  "https://www.gutenberg.org/cache/epub/44562/pg44562.txt", #A Dictionary of Arts, Manufactures and Mines by Andrew Ure
  "https://www.gutenberg.org/cache/epub/10766/pg10766.txt", #Enquire within upon everything by Robert Kemp Philp
  "https://www.gutenberg.org/cache/epub/21682/pg21682.txt", #The field and garden vegetables of America : containing full descriptions of…
  "https://www.gutenberg.org/cache/epub/5402/pg5402.txt",   #1811 Dictionary of the Vulgar Tongue by Francis Grose
  "https://www.gutenberg.org/cache/epub/47018/pg47018.txt", #A Dictionary of Slang, Cant, and Vulgar Words by John Camden Hotten
  "https://www.gutenberg.org/cache/epub/42108/pg42108.txt",   #The slang dictionary : Etymological, historical and anecdotal by Hotten
  "https://www.gutenberg.org/cache/epub/45249/pg45249.txt", #The History of Signboards, from the Earliest times to the Present Day by Larwood et al.
  "https://www.gutenberg.org/cache/epub/44028/pg44028.txt", #The Every Day Book of History and Chronology by Joel Munsell
  "https://www.gutenberg.org/cache/epub/70718/pg70718.txt", #The inter ocean curiosity shop for the year 1883 by Various
  "https://www.gutenberg.org/cache/epub/747/pg747.txt",     #Anomalies and Curiosities of Medicine by George M. Gould and Walter L. Pyle
  "https://www.gutenberg.org/cache/epub/39733/pg39733.txt", #Cooley's Cyclopædia of Practical Receipts and Collateral Information in the…
  "https://www.gutenberg.org/cache/epub/41957/pg41957.txt", #Cooley's Cyclopædia of Practical Receipts and Collateral Information in the…
  "https://www.gutenberg.org/cache/epub/26535/pg26535.txt", #The Standard Electrical Dictionary by T. O'Conor Sloane
  "https://www.gutenberg.org/cache/epub/12513/pg12513.txt", #A Classification and Subject Index for Cataloguing and Arranging the Books and…
  "https://www.gutenberg.org/cache/epub/61526/pg61526.txt", #A Dictionary of Islam by Thomas Patrick Hughes
  "https://www.gutenberg.org/cache/epub/47682/pg47682.txt", #American Book-Plates: A Guide to Their Study with Examples by Charles Dexter Allen
  "https://www.gutenberg.org/cache/epub/28900/pg28900.txt", #English Synonyms and Antonyms by James Champlin Fernald
  "https://www.gutenberg.org/cache/epub/41697/pg41697.txt", #Merck's 1899 Manual of the Materia Medica by Merck & Co.
  "https://www.gutenberg.org/cache/epub/41140/pg41140.txt", #Bible Atlas: A Manual of Biblical Geography and History by Jesse Lyman Hurlbut
  "https://www.gutenberg.org/cache/epub/57705/pg57705.txt", #An Universal Dictionary of the Marine by William Falconer
  "https://www.gutenberg.org/cache/epub/59959/pg59959.txt", #English Surnames: Their Sources and Significations by Bardsley
  "https://www.gutenberg.org/cache/epub/39284/pg39284.txt", #Curiosities of Puritan Nomenclature by Charles Wareing Endell Bardsley
  "https://www.gutenberg.org/cache/epub/51739/pg51739.txt", #The Romance of the London Directory by Charles Wareing Endell Bardsley
  "https://www.gutenberg.org/cache/epub/74739/pg74739.txt", #A military dictionary : or, Explanation of the several systems of discipline…
  "https://www.gutenberg.org/cache/epub/62250/pg62250.txt", #History, Gazetteer, and Directory of Shropshire [1851] by Samuel Bagshaw
  "https://www.gutenberg.org/cache/epub/71324/pg71324.txt", #A dictionary of American authors by Oscar Fay Adams
  "https://www.gutenberg.org/cache/epub/37072/pg37072.txt", #A Brief Handbook of English Authors by Oscar Fay Adams
  "https://www.gutenberg.org/cache/epub/78263/pg78263.txt", #Slang and its analogues past and present, volume 1 [of 7] : A dictionary,…
  "https://www.gutenberg.org/cache/epub/77292/pg77292.txt", #Slang and its analogues past and present, volume 2 [of 7] : A dictionary,…
  "https://www.gutenberg.org/cache/epub/74870/pg74870.txt", #Slang and its analogues past and present, volume 3 [of 7] : A dictionary,…
  "https://www.gutenberg.org/cache/epub/79663/pg79663.txt", #Slang and its analogues past and present, volume 4 [of 7] : A dictionary,…
  "https://www.gutenberg.org/cache/epub/68769/pg68769.txt", #A classical dictionary by John Lemprière
  "https://www.gutenberg.org/cache/epub/59563/pg59563.txt", #A Military Dictionary and Gazetteer by Thomas Wilhelm
  "https://www.gutenberg.org/cache/epub/42172/pg42172.txt", #Every Boy's Book: A Complete Encyclopædia of Sports and Amusements by Routledge
  "https://www.gutenberg.org/cache/epub/70427/pg70427.txt", #History for ready reference, Volumes 1 to 5 by J. N. Larned
  "https://www.gutenberg.org/cache/epub/14472/pg14472.txt", #Showell's Dictionary of Birmingham by Thomas T. Harman and Walter Showell
  "https://www.gutenberg.org/cache/epub/51104/pg51104.txt", #Allen & Morton's West-London Directory for 1867 by Allen & Morton
  "https://www.gutenberg.org/cache/epub/38390/pg38390.txt", #A Dictionary of English Synonymes and Synonymous or Parallel Expressions by Soule
  "https://www.gutenberg.org/cache/epub/67629/pg67629.txt", #An Illustrated Dictionary of Words used in Art and Archaeology by John W. Mollett
  "https://www.gutenberg.org/cache/epub/34513/pg34513.txt", #A Biographical Dictionary of Freethinkers of All Ages and Nations by J. M. Wheeler
  "https://www.gutenberg.org/cache/epub/52930/pg52930.txt", #Names: and Their Meaning; A Book for the Curious by Leopold Wagner
  "https://www.gutenberg.org/cache/epub/14091/pg14091.txt", #Barkham Burroughs' Encyclopaedia of Astounding Facts and Useful Information,…
  "https://www.gutenberg.org/cache/epub/53884/pg53884.txt", #A Biblical and Theological Dictionary by Richard Watson
  "https://www.gutenberg.org/cache/epub/70957/pg70957.txt", #The treasury of languages : A rudimentary dictionary of universal philology
  "https://www.gutenberg.org/cache/epub/65909/pg65909.txt", #A Smaller Dictionary of Greek and Roman Antiquities by William Smith
  "https://www.gutenberg.org/cache/epub/2096/pg2096.txt",   #A Smaller History of Greece: from the Earliest Times to the Roman Conquest by Smith
  "https://www.gutenberg.org/cache/epub/19694/pg19694.txt", #A Smaller History of Rome by William Smith and Eugene Lawrence
  "https://www.gutenberg.org/cache/epub/71502/pg71502.txt", #American politics (non-partisan) from the beginning to date : embodying…
  "https://www.gutenberg.org/cache/epub/15191/pg15191.txt", #The Commercial Products of the Vegetable Kingdom by P. L. Simmonds
  "https://www.gutenberg.org/cache/epub/71509/pg71509.txt", #The curiosities of food : or, The dainties and delicacies of different nations…
  "https://www.gutenberg.org/cache/epub/57340/pg57340.txt", #Coffee and chicory : Their culture, chemical composition, preparation for…
  "https://www.gutenberg.org/cache/epub/49912/pg49912.txt", #The Employments of Women: A Cyclopædia of Woman's Work by Virginia Penny
  "https://www.gutenberg.org/cache/epub/52320/pg52320.txt", #Vocabulum; or The Rogue's Lexicon by George W. Matsell
  "https://www.gutenberg.org/cache/epub/64006/pg64006.txt", #Rogers' Directory of Norwich and Neighbourhood by Edmund Dawson Rogers
  "https://www.gutenberg.org/cache/epub/76850/pg76850.txt", #The watchmakers' hand book : intended as a workshop companion for those…
  "https://www.gutenberg.org/cache/epub/53336/pg53336.txt", #The Public School Word-book by John Stephen Farmer
  "https://www.gutenberg.org/cache/epub/30323/pg30323.txt", #The Book of Religions by John Hayward
  "https://www.gutenberg.org/cache/epub/49791/pg49791.txt", #Miscellanea Curiosa, Vol. 1 by Edmond Halley et al.
  "https://www.gutenberg.org/cache/epub/43991/pg43991.txt", #A Hand-book to the Primates, Volume 1 (of 2) by Henry O. Forbes
  "https://www.gutenberg.org/cache/epub/43992/pg43992.txt", #A Hand-book to the Primates, Volume 2 (of 2) by Henry O. Forbes
  "https://www.gutenberg.org/cache/epub/75040/pg75040.txt", #The wonders of the world : a complete museum, descriptive and pictorial, of…
  "https://www.gutenberg.org/cache/epub/14403/pg14403.txt", #1001 Questions and Answers on Orthography and Reading by B. A. Hathaway
  "https://www.gutenberg.org/cache/epub/45809/pg45809.txt", #A Glossary of Words used in the Country of Wiltshire by Dartnell and Goddard
  "https://www.gutenberg.org/cache/epub/59215/pg59215.txt", #Rules for a Dictionary Catalogue by Charles A. Cutter
  "https://www.gutenberg.org/cache/epub/58160/pg58160.txt", #Allen's West London Street Directory, 1868 by Samuel Allen
  "https://www.gutenberg.org/cache/epub/38457/pg38457.txt", #Alden's Handy Atlas of the World by John B. Alden
  "https://www.gutenberg.org/cache/epub/60729/pg60729.txt", #Slater's [1859] Shropshire Directory by I. Slater
  "https://www.gutenberg.org/cache/epub/57783/pg57783.txt", #The Legislative Manual, of the State of Colorado by Thomas B. Corbett
  "https://www.gutenberg.org/cache/epub/28413/pg28413.txt", #The Student's Companion to Latin Authors by George Middleton and Thomas Ross Mills
  "https://www.gutenberg.org/cache/epub/25212/pg25212.txt", #A Glossary of Provincial Words & Phrases in use in Somersetshire by Williams
  "https://www.gutenberg.org/cache/epub/56162/pg56162.txt", #The Illustrated Dictionary of Gardening, Division 1; A to Car. by George Nicholson et al.
  "https://www.gutenberg.org/cache/epub/14968/pg14968.txt", #The Standard Operas: Their Plots, Their Music, and Their Composers by Upton
  "https://www.gutenberg.org/cache/epub/22793/pg22793.txt", #The Standard Oratorios: Their Stories, Their Music, And Their Composers by Upton
  "https://www.gutenberg.org/cache/epub/32248/pg32248.txt", #The Standard Cantatas: Their Stories, Their Music, and Their Composers by Upton
  "https://www.gutenberg.org/cache/epub/67827/pg67827.txt", #Life of Haydn by Ludwig Nohl
  "https://www.gutenberg.org/cache/epub/36734/pg36734.txt", #The Browning Cyclopædia: A Guide to the Study of the Works of Robert Browning
  "https://www.gutenberg.org/cache/epub/59821/pg59821.txt", #Handbook of the new Library of Congress by Small, Caffin, and Spofford
  "https://www.gutenberg.org/cache/epub/44460/pg44460.txt", #Carleton's Condensed Classical Dictionary by George Washington Carleton
  "https://www.gutenberg.org/cache/epub/23852/pg23852.txt", #Canadian Postal Guide by Canada. Post Office Department
  "https://www.gutenberg.org/cache/epub/53084/pg53084.txt", #A manual of Italian literature by Francis Henry Cliffe
  "https://www.gutenberg.org/cache/epub/63912/pg63912.txt", #The American Encyclopedia of History, Biography and Travel by Thomas H. Prescott
  "https://www.gutenberg.org/cache/epub/54233/pg54233.txt", #The Vivisectors' Directory by Benjamin Bryan and Frances Power Cobbe
  "https://www.gutenberg.org/cache/epub/62386/pg62386.txt", #Melville and Co.'s Directory and Gazetteer of Norwich, Yarmouth, Dereham,…
  "https://www.gutenberg.org/cache/epub/32829/pg32829.txt", #A concise history and directory of the city of Norwich for 1811 by Berry
  "https://www.gutenberg.org/cache/epub/58200/pg58200.txt", #Lakeland Words by Bryham Kirkby
  "https://www.gutenberg.org/cache/epub/17265/pg17265.txt", #Companion to the Bible by E. P. Barrows
  "https://www.gutenberg.org/cache/epub/62401/pg62401.txt", #Mason's Norwich General and Commercial Directory & Handbook by R. Hindry Mason
  "https://www.gutenberg.org/cache/epub/49524/pg49524.txt", #Knowledge for the Time by John Timbs
  "https://www.gutenberg.org/cache/epub/48516/pg48516.txt", #Curiosities of Science, Past and Present by John Timbs
  "https://www.gutenberg.org/cache/epub/61084/pg61084.txt", #The Standard Manual for Baptist Churches by Edward T. Hiscox
  "https://www.gutenberg.org/cache/epub/12864/pg12864.txt", #A Collection of College Words and Customs by Benjamin Homer Hall
  "https://www.gutenberg.org/cache/epub/57898/pg57898.txt", #Useful Knowledge: Volume 1. Minerals by William Bingley
  "https://www.gutenberg.org/cache/epub/57954/pg57954.txt", #Useful Knowledge: Volume 2. Vegetables by William Bingley
  "https://www.gutenberg.org/cache/epub/58118/pg58118.txt", #Useful Knowledge: Volume 3. Animals by William Bingley
  "https://www.gutenberg.org/cache/epub/60126/pg60126.txt", #Pens and types : or, Hints and helps for those who write, print, read, teach,…
  "https://www.gutenberg.org/cache/epub/70210/pg70210.txt", #A select glossary of English words used formerly in senses different from their…
  "https://www.gutenberg.org/cache/epub/8163/pg8163.txt",   #Handbook of Universal Literature, From the Best and Latest Authorities by Botta
  "https://www.gutenberg.org/cache/epub/17712/pg17712.txt", #The moon : a full description and map of its principal physical features by Elger
  "https://www.gutenberg.org/cache/epub/2468/pg2468.txt",   #Repertory of The Comedie Humaine, Part 1 by Cerfberr and Christophe
  "https://www.gutenberg.org/cache/epub/47367/pg47367.txt", #British Pomology by Robert Hogg
  "https://www.gutenberg.org/cache/epub/65002/pg65002.txt", #The Birds of Australia, Vol. 1 of 7 by John Gould
  "https://www.gutenberg.org/cache/epub/60302/pg60302.txt", #The Birds of Australia, Vol. 2 of 7 by John Gould
  "https://www.gutenberg.org/cache/epub/60646/pg60646.txt", #The Birds of Australia, Vol. 3 of 7 by John Gould
  "https://www.gutenberg.org/cache/epub/60833/pg60833.txt", #The Birds of Australia, Vol. 4 of 7 by John Gould
  "https://www.gutenberg.org/cache/epub/62524/pg62524.txt", #The Birds of Australia, Vol. 5 of 7 by John Gould
  "https://www.gutenberg.org/cache/epub/52232/pg52232.txt", #A Bible Hand-Book by F. C. Holliday
  "https://www.gutenberg.org/cache/epub/71466/pg71466.txt", #Queer questions and ready replies : A collection of four hundred questions in…
  "https://www.gutenberg.org/cache/epub/70907/pg70907.txt", #Catalogue of the dipterous insects collected at Singapore and Malacca by Wallace
  "https://www.gutenberg.org/cache/epub/59076/pg59076.txt", #The Sporting Dictionary, and Rural Repository, Volume 1 (of 2) by William Taplin
  "https://www.gutenberg.org/cache/epub/59291/pg59291.txt", #The Sporting Dictionary and Rural Repository, Volume 2 (of 2) by William Taplin
  "https://www.gutenberg.org/cache/epub/75849/pg75849.txt", #Aggravating ladies : Being a list of works published under the pseudonym of "a…
  "https://www.gutenberg.org/cache/epub/9916/pg9916.txt",   #Spalding's Baseball Guide and Official League Book for 1895 by Henry Chadwick
  "https://www.gutenberg.org/cache/epub/65710/pg65710.txt", #A Synopsis of the British Mosses by Charles C. P. Hobkirk
  "https://www.gutenberg.org/cache/epub/31165/pg31165.txt", #The Church Handy Dictionary by Anonymous
  "https://www.gutenberg.org/cache/epub/15484/pg15484.txt", #The Care and Feeding of Children by L. Emmett Holt
  "https://www.gutenberg.org/cache/epub/14551/pg14551.txt", #A catechism of Christian doctrine, no. 1 : for first communion classes
  "https://www.gutenberg.org/cache/epub/14552/pg14552.txt", #A catechism of Christian doctrine, no. 2 : for confirmation classes by De Concilio et al.
  "https://www.gutenberg.org/cache/epub/14553/pg14553.txt", #A catechism of Christian doctrine, no. 3 : for two years' course for post-…
  "https://www.gutenberg.org/cache/epub/16728/pg16728.txt", #A Catechism of Familiar Things; by Anonymous
  "https://www.gutenberg.org/cache/epub/76379/pg76379.txt", #Catechism of the locomotive by Matthias N. Forney and Georg Kosak
  "https://www.gutenberg.org/cache/epub/10998/pg10998.txt", #A Catechism of the Steam Engine by C. E. John Bourne
  "https://www.gutenberg.org/cache/epub/59133/pg59133.txt", #A Body of Divinity, Vol. 1 (of 4) by Thomas Ridgley
  "https://www.gutenberg.org/cache/epub/62149/pg62149.txt", #A Body of Divinity, Vol. 2 (of 4) by Thomas Ridgley
  "https://www.gutenberg.org/cache/epub/64185/pg64185.txt", #A Body of Divinity, Vol. 3 (of 4) by Thomas Ridgley
  "https://www.gutenberg.org/cache/epub/65484/pg65484.txt", #A Body of Divinity, Vol. 4 (of 4) by Thomas Ridgley
  "https://www.gutenberg.org/cache/epub/69376/pg69376.txt", #Anti-slavery catechism by Lydia Maria Child
  "https://www.gutenberg.org/cache/epub/39293/pg39293.txt", #The Gentlemen's Book of Etiquette and Manual of Politeness by Cecil B. Hartley
  "https://www.gutenberg.org/cache/epub/23860/pg23860.txt", #The Young Man's Guide by William A. Alcott
  "https://www.gutenberg.org/cache/epub/27394/pg27394.txt", #A New Guide for Emigrants to the West by John Mason Peck
  "https://www.gutenberg.org/cache/epub/14400/pg14400.txt", #Manual of Egyptian Archaeology and Guide to the Study of Antiquities in Egypt
  "https://www.gutenberg.org/cache/epub/57630/pg57630.txt", #Spons' Household Manual by E. & F. N. Spon
  "https://www.gutenberg.org/cache/epub/69487/pg69487.txt", #The complete servant by servant Samuel Adams and Sarah Adams
  "https://www.gutenberg.org/cache/epub/26597/pg26597.txt", #How to Behave: A Pocket Manual of Republican Etiquette, and Guide to Correct…
  "https://www.gutenberg.org/cache/epub/6598/pg6598.txt",   #American Woman's Home: Or, Principles of Domestic Science; by Beecher and Stowe
  "https://www.gutenberg.org/cache/epub/21829/pg21829.txt", #A Treatise on Domestic Economy; For the Use of Young Ladies at Home and at…
  "https://www.gutenberg.org/cache/epub/4337/pg4337.txt",   #Power Through Repose by Annie Payson Call
  "https://www.gutenberg.org/cache/epub/4385/pg4385.txt",   #As a Matter of Course by Annie Payson Call
  "https://www.gutenberg.org/cache/epub/41279/pg41279.txt", #The Butterfly Book by W. J. Holland
  "https://www.gutenberg.org/cache/epub/43614/pg43614.txt", #The Valleys of Tirol: Their traditions and customs and how to visit them by Busk
  "https://www.gutenberg.org/cache/epub/19924/pg19924.txt", #Plain Facts for Old and Young by John Harvey Kellogg
  "https://www.gutenberg.org/cache/epub/17367/pg17367.txt", #First Book in Physiology and Hygiene by John Harvey Kellogg
  "https://www.gutenberg.org/cache/epub/74519/pg74519.txt", #The uses of water in health and disease : A practical treatise on the bath,…
  "https://www.gutenberg.org/cache/epub/13923/pg13923.txt", #The Whitehouse Cookbook (1887) by F. L. Gillette and Hugo Ziemann
  "https://www.gutenberg.org/cache/epub/20889/pg20889.txt", #Facts About Champagne and Other Sparkling Wines by Henry Vizetelly
  "https://www.gutenberg.org/cache/epub/46953/pg46953.txt", #A History of Champagne, with Notes on the Other Sparkling Wines of France
  "https://www.gutenberg.org/cache/epub/57168/pg57168.txt", #The Land Beyond the Forest: Facts, Figures, and Fancies from Transylvania by Gerard
  "https://www.gutenberg.org/cache/epub/41625/pg41625.txt", #Curious Facts in the History of Insects; Including Spiders and Scorpions. by Cowan
  "https://www.gutenberg.org/cache/epub/51034/pg51034.txt", #The Invention of Printing. by Theodore Low De Vinne
  "https://www.gutenberg.org/cache/epub/35123/pg35123.txt", #The Ladies' Book of Etiquette, and Manual of Politeness by Florence Hartley
  "https://www.gutenberg.org/cache/epub/50750/pg50750.txt", #The Officer's Manual: Napoleon's Maxims of War by Emperor of the French Napoleon I
  "https://www.gutenberg.org/cache/epub/46254/pg46254.txt", #Mrs. Hale's Receipts for the Million by Sarah Josepha Buell Hale
  "https://www.gutenberg.org/cache/epub/75494/pg75494.txt", #The workwoman's guide : containing instructions to the inexperienced in…
  "https://www.gutenberg.org/cache/epub/9097/pg9097.txt",   #Robert's Rules of Order by Henry M. Robert
  "https://www.gutenberg.org/cache/epub/42709/pg42709.txt", #Poisons, Their Effects and Detection by Alexander Wynter Blyth
  "https://www.gutenberg.org/cache/epub/77516/pg77516.txt", #Astronomy by Agnes M. Clerke, A. Fowler, and J. Ellard Gore
  "https://www.gutenberg.org/cache/epub/40383/pg40383.txt", #Musical Myths and Facts, Volume 1 (of 2) by Carl Engel
  "https://www.gutenberg.org/cache/epub/40384/pg40384.txt", #Musical Myths and Facts, Volume 2 (of 2) by Carl Engel
  "https://www.gutenberg.org/cache/epub/45124/pg45124.txt", #Great Facts by Frederick C. Bakewell
  "https://www.gutenberg.org/cache/epub/23776/pg23776.txt", #The School of Recreation (1684 edition) by R. H.
  "https://www.gutenberg.org/cache/epub/17727/pg17727.txt", #The School of Recreation (1696 edition) by R. H.
  "https://www.gutenberg.org/cache/epub/1697/pg1697.txt",   #Madam How and Lady Why; Or, First Lessons in Earth Lore for Children by Kingsley
  "https://www.gutenberg.org/cache/epub/695/pg695.txt",     #Glaucus; Or, The Wonders of the Shore by Charles Kingsley
  "https://www.gutenberg.org/cache/epub/17437/pg17437.txt", #Health and Education by Charles Kingsley
  "https://www.gutenberg.org/cache/epub/67257/pg67257.txt", #Lessons in Sabre, Singlestick, Sabre & Bayonet, and Sword Feats by J. M. Waite
  "https://www.gutenberg.org/cache/epub/54372/pg54372.txt", #The Royal Pastime of Cock-fighting by R. H.
  "https://www.gutenberg.org/cache/epub/28681/pg28681.txt", #The Cook's Oracle; and Housekeeper's Manual by William Kitchiner
  "https://www.gutenberg.org/cache/epub/40891/pg40891.txt", #The Art of Invigorating and Prolonging Life by William Kitchiner
  "https://www.gutenberg.org/cache/epub/70726/pg70726.txt", #The traveller's oracle; or, maxims for locomotion, part 2 (of 2) : Containing…
  "https://www.gutenberg.org/cache/epub/25520/pg25520.txt", #The Stock-Feeder's Manual by Sir Charles Alexander Cameron
  "https://www.gutenberg.org/cache/epub/41595/pg41595.txt", #The Funny Side of Physic by Addison Darre Crabtre
  "https://www.gutenberg.org/cache/epub/34350/pg34350.txt", #A Manual of Elementary Geology by Sir Charles Lyell
  "https://www.gutenberg.org/cache/epub/3772/pg3772.txt",   #The Student's Elements of Geology by Sir Charles Lyell
  "https://www.gutenberg.org/cache/epub/60687/pg60687.txt", #The Magician's Own Book, or, the Whole Art of Conjuring by Arnold and Cahil
  "https://www.gutenberg.org/cache/epub/14444/pg14444.txt", #The Complete English Tradesman (1839 ed.) by Daniel Defoe
  "https://www.gutenberg.org/cache/epub/10726/pg10726.txt", #Outlines of Lessons in Botany, Part I; from Seed to Leaf by Jane Newell Moore
  "https://www.gutenberg.org/cache/epub/49155/pg49155.txt", #Marion Harland's Cookery for Beginners by Marion Harland
  "https://www.gutenberg.org/cache/epub/46144/pg46144.txt", #Six Cups of Coffee by Parloa, Campbell, Corson, Harland, Lincoln, Owen, and Poole
  "https://www.gutenberg.org/cache/epub/16800/pg16800.txt", #The Secret of a Happy Home (1896) by Marion Harland
  "https://www.gutenberg.org/cache/epub/49958/pg49958.txt", #The Dinner Year-Book by Marion Harland
  "https://www.gutenberg.org/cache/epub/49944/pg49944.txt", #Breakfast, Luncheon and Tea by Marion Harland
  "https://www.gutenberg.org/cache/epub/48804/pg48804.txt", #Common Sense in the Household: A Manual of Practical Housewifery by Marion Harland
  "https://www.gutenberg.org/cache/epub/11722/pg11722.txt", #First Lessons in Geography by James Monteith
  "https://www.gutenberg.org/cache/epub/58166/pg58166.txt", #The Art of Tying the Cravat; Demonstrated in sixteen lessons by H. Le Blanc
  "https://www.gutenberg.org/cache/epub/77859/pg77859.txt", #The starry skies : or, First lessons on the sun, moon and stars by Agnes Giberne
  "https://www.gutenberg.org/cache/epub/62871/pg62871.txt", #Thoughts on a Pebble, or, A First Lesson in Geology by Gideon Algernon Mantell
  "https://www.gutenberg.org/cache/epub/53778/pg53778.txt", #First Lessons in Natural Philosophy for Beginners by Joseph C. Martindale
  "https://www.gutenberg.org/cache/epub/75832/pg75832.txt", #First lessons in the principles of cooking by Lady Barker
  "https://www.gutenberg.org/cache/epub/71943/pg71943.txt", #Popular lessons in astronomy, on a new plan : in which some of the leading…
  "https://www.gutenberg.org/cache/epub/50903/pg50903.txt", #Sleight of Hand: A Practical Manual of Legerdemain for Amateurs & Others by Sachs
  "https://www.gutenberg.org/cache/epub/43417/pg43417.txt", #A conchological manual by G. B. Sowerby
  "https://www.gutenberg.org/cache/epub/24583/pg24583.txt", #Langstroth on the Hive and the Honey-Bee: A Bee Keeper's Manual by L. L. Langstroth
  "https://www.gutenberg.org/cache/epub/14594/pg14594.txt", #Cassell's Vegetarian Cookery: A Manual of Cheap and Wholesome Diet by A. G. Payne
  "https://www.gutenberg.org/cache/epub/61112/pg61112.txt", #A New Light of Alchymie by Michal Sedziwój and Paracelsus
  "https://www.gutenberg.org/cache/epub/68043/pg68043.txt", #Fossil plants, Vol. 1 : [A text-book] for students of botany and geology by Seward
  "https://www.gutenberg.org/cache/epub/24564/pg24564.txt", #A Manual of the Operations of Surgery by Joseph Bell
  "https://www.gutenberg.org/cache/epub/12327/pg12327.txt", #The Jewish Manual by Lady Judith Cohen Montefiore
  "https://www.gutenberg.org/cache/epub/62988/pg62988.txt", #The Complete Manual of Catholic Piety by William Gahan
  "https://www.gutenberg.org/cache/epub/71928/pg71928.txt", #The gunsmith's manual : A complete handbook for the American gunsmith, being…
  "https://www.gutenberg.org/cache/epub/71463/pg71463.txt", #Hanky Panky : A book of conjuring tricks by Wiljalba Frikell
  "https://www.gutenberg.org/cache/epub/44509/pg44509.txt", #A Manual of the Historical Development of Art by G. G. Zerffi
  "https://www.gutenberg.org/cache/epub/77729/pg77729.txt", #The kedge-anchor : or, Young sailors' assistant. by William N. Brady
  "https://www.gutenberg.org/cache/epub/68388/pg68388.txt", #How to thought-read : A manual of instruction in the strange and mystic in…
  "https://www.gutenberg.org/cache/epub/51204/pg51204.txt", #Corticelli Home Needlework, 1898: A Manual of Art, Embroidery and Knitting
  "https://www.gutenberg.org/cache/epub/61065/pg61065.txt", #Modern American Drinks: How to Mix and Serve All Kinds of Cups and Drinks
  "https://www.gutenberg.org/cache/epub/58452/pg58452.txt", #Fences, Gates and Bridges: A Practical Manual by George A. Martin
  "https://www.gutenberg.org/cache/epub/50059/pg50059.txt", #How to Behave and How to Amuse: A Handy Manual of Etiquette and Parlor Games
  "https://www.gutenberg.org/cache/epub/44096/pg44096.txt", #An elementary manual of New Zealand entomology : being an introduction to the…
  "https://www.gutenberg.org/cache/epub/44551/pg44551.txt", #New Zealand Moths and Butterflies (Macro-Lepidoptera) by G. V. Hudson
  "https://www.gutenberg.org/cache/epub/31605/pg31605.txt", #The Cooking Manual of Practical Directions for Economical Every-Day Cookery
  "https://www.gutenberg.org/cache/epub/39747/pg39747.txt", #A Manual of Ancient History by A. H. L. Heeren
  "https://www.gutenberg.org/cache/epub/68394/pg68394.txt", #The Trouvelot astronomical drawings manual by E. L. Trouvelot
  "https://www.gutenberg.org/cache/epub/71233/pg71233.txt", #Anthropology : an introduction to the study of man and civilization by Tylor
  "https://www.gutenberg.org/cache/epub/47836/pg47836.txt", #The Matron's Manual of Midwifery, and the Diseases of Women During Pregnancy…
  "https://www.gutenberg.org/cache/epub/50100/pg50100.txt", #How to Do Chemical Tricks by active 1894-1902 A. Anderson
  "https://www.gutenberg.org/cache/epub/64054/pg64054.txt", #Cameo Cutting by John B. Marsh
  "https://www.gutenberg.org/cache/epub/35329/pg35329.txt", #A Manual of the Antiquity of Man by J. P. MacLean
  "https://www.gutenberg.org/cache/epub/42982/pg42982.txt", #Photography in the studio and in the field : a practical manual designed as…
  "https://www.gutenberg.org/cache/epub/57979/pg57979.txt", #Mother, Nurse and Infant by S. P. Sackett
  "https://www.gutenberg.org/cache/epub/39500/pg39500.txt", #Beauties and Antiquities of Ireland by Thomas O'Neill Russell
  "https://www.gutenberg.org/cache/epub/30975/pg30975.txt", #Soil Culture by J. H. Walden
  "https://www.gutenberg.org/cache/epub/57320/pg57320.txt", #Telescopic Work for Starlight Evenings by William F. Denning
  "https://www.gutenberg.org/cache/epub/31006/pg31006.txt", #The Author's Printing and Publishing Assistant by Frederick Saunders
  "https://www.gutenberg.org/cache/epub/23476/pg23476.txt", #Churchwardens' Manual by George Henry Sumner
  "https://www.gutenberg.org/cache/epub/10733/pg10733.txt", #Studies in Civics by J. T. McCleary
  "https://www.gutenberg.org/cache/epub/57612/pg57612.txt", #Ladies' manual of art; or, profit and pastime. by Anonymous
  "https://www.gutenberg.org/cache/epub/29248/pg29248.txt", #The Young Lady's Equestrian Manual by Anonymous
  "https://www.gutenberg.org/cache/epub/60757/pg60757.txt", #Manual of Parliamentary Practice by Luther Stearns Cushing
  "https://www.gutenberg.org/cache/epub/62609/pg62609.txt", #Locomotive Engine Running and Management by Angus Sinclair
  "https://www.gutenberg.org/cache/epub/76936/pg76936.txt", #A practical guide for making post-mortem examinations : and for the study of…
  "https://www.gutenberg.org/cache/epub/63710/pg63710.txt", #A Manual of Photographic Chemistry, Including the Practice of the Collodion…
  "https://www.gutenberg.org/cache/epub/39044/pg39044.txt", #Advice to the people in general, with regard to their health by S. A. D. Tissot
  "https://www.gutenberg.org/cache/epub/49077/pg49077.txt", #The Art of Horse-Shoeing: A Manual for Farriers by William Hunting
  "https://www.gutenberg.org/cache/epub/38831/pg38831.txt", #The Barb and the Bridle by active 1866-1874 Robert Henderson
  "https://www.gutenberg.org/cache/epub/37142/pg37142.txt", #How to Care for the Insane: A Manual for Nurses by William D. Granger
  "https://www.gutenberg.org/cache/epub/44604/pg44604.txt", #How to become an engineer : containing full instructions how to proceed in…
  "https://www.gutenberg.org/cache/epub/74523/pg74523.txt", #A church dictionary by Walter Farquhar Hook
  "https://www.gutenberg.org/cache/epub/37309/pg37309.txt", #Violins and Violin Makers by Joseph Pearce
  "https://www.gutenberg.org/cache/epub/74485/pg74485.txt", #A handy dictionary of mythology : For everyday readers by Thomas Preston
  "https://www.gutenberg.org/cache/epub/44040/pg44040.txt", #Worcestershire in the Nineteenth Century by T. C. Turberville
  "https://www.gutenberg.org/cache/epub/67768/pg67768.txt", #Hand-book of Sanitary Information for Householders by Roger Sherman Tracy
  "https://www.gutenberg.org/cache/epub/64140/pg64140.txt", #Soup and Soup Making by Emma P. Ewing
  "https://www.gutenberg.org/cache/epub/78002/pg78002.txt", #The ladies' hand-book of millinery and dressmaking : With plain instructions…
  "https://www.gutenberg.org/cache/epub/66524/pg66524.txt", #The Wine Press and the Cellar: A Manual for the Wine-Maker and the Cellar-Man
  "https://www.gutenberg.org/cache/epub/58756/pg58756.txt", #A manual of pyrotechny : or, A familiar system of recreative fire-works
  "https://www.gutenberg.org/cache/epub/60550/pg60550.txt", #The Science of Brickmaking by George Frederick Harris
  "https://www.gutenberg.org/cache/epub/67942/pg67942.txt", #The honey bee: a manual of instruction in apiculture by Frank Benton
  "https://www.gutenberg.org/cache/epub/68157/pg68157.txt", #Manual of the apiary by Albert John Cook
  "https://www.gutenberg.org/cache/epub/32838/pg32838.txt", #The Canadian Curler's Manual by James Bicket
  "https://www.gutenberg.org/cache/epub/29701/pg29701.txt", #The Mother's Manual of Children's Diseases by Charles West
  "https://www.gutenberg.org/cache/epub/63517/pg63517.txt", #A Practical Manual of the Collodion Process, Giving in Detail a Method For…
  "https://www.gutenberg.org/cache/epub/71782/pg71782.txt", #Steel : a manual for steel users by William Metcalf
  "https://www.gutenberg.org/cache/epub/52850/pg52850.txt", #Barbers' Manual (Part 1); Text Book on Taxidermy (Part 2) by T. J. McConnaughay
  "https://www.gutenberg.org/cache/epub/76043/pg76043.txt", #Every woman her own flower gardener : A handy manual of flower gardening for…
  "https://www.gutenberg.org/cache/epub/62200/pg62200.txt", #Sabbath Defence Tactics: a manual by James Bridges
  "https://www.gutenberg.org/cache/epub/12942/pg12942.txt", #The Highland Fling and How to Teach it by Horatio N. Grant
  "https://www.gutenberg.org/cache/epub/67477/pg67477.txt", #Bee-keeping for the Many; or, The management of the common and Ligurian honey…
  "https://www.gutenberg.org/cache/epub/67439/pg67439.txt", #The Bee Keeper's Guide, Third Edition by J. H. Payne
  "https://www.gutenberg.org/cache/epub/57273/pg57273.txt", #The Stocking-Knitter's Manual: A Handy Book for the Work-Table by Cupples
  "https://www.gutenberg.org/cache/epub/73434/pg73434.txt", #Prof. E. H. Crane's manual of instructions to undertakers : With full and…
  "https://www.gutenberg.org/cache/epub/78603/pg78603.txt", #The harness makers' illustrated manual by William N. Fitz-Gerald
  ##### LARGER BOOKS/TEXTBOOKS #####
  "https://www.gutenberg.org/cache/epub/25717/pg25717.txt", #The History of the Decline and Fall of the Roman Empire by Edward Gibbon
  "https://www.gutenberg.org/cache/epub/2464/pg2464.txt",   #History of Florence and of the Affairs of Italy by Niccolò Machiavelli
  "https://www.gutenberg.org/cache/epub/25848/pg25848.txt", #William the Conqueror by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/27551/pg27551.txt", #Hannibal by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/26939/pg26939.txt", #Richard I by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/26734/pg26734.txt", #Charles I by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/27802/pg27802.txt", #Darius the Great by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/28433/pg28433.txt", #Richard II by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/25351/pg25351.txt", #Xerxes by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/27240/pg27240.txt", #Pyrrhus by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/28283/pg28283.txt", #Mary Queen of Scots by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/15400/pg15400.txt", #Renaissance in Italy, Volume 1 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/41924/pg41924.txt", #Renaissance in Italy, Volume 2 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/11559/pg11559.txt", #Renaissance in Italy, Volume 3 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/35792/pg35792.txt", #Renaissance in Italy, Volume 4 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/36448/pg36448.txt", #Renaissance in Italy, Volume 5 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/16504/pg16504.txt", #Renaissance in Italy, Volumes 6 and 7 (of 7) by John Addington Symonds
  "https://www.gutenberg.org/cache/epub/30624/pg30624.txt", #Alexander the Great by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/28667/pg28667.txt", #Genghis Khan, Makers of History Series by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/27692/pg27692.txt", #Romulus by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/27312/pg27312.txt", #Nero by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/28561/pg28561.txt", #Richard III by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/40205/pg40205.txt", #History of Cleopatra, Queen of Egypt by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/29107/pg29107.txt", #Queen Elizabeth by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/16545/pg16545.txt", #King Alfred of England by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/30707/pg30707.txt", #Cyrus the Great by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/21889/pg21889.txt", #Peter the Great by Jacob Abbott
  "https://www.gutenberg.org/cache/epub/1972/pg1972.txt",   #History of the Britons (Historia Brittonum) by active 796 Nennius
  "https://www.gutenberg.org/cache/epub/61475/pg61475.txt", #Byzantine Constantinople, the walls of the city and adjoining historical sites
  "https://www.gutenberg.org/cache/epub/45815/pg45815.txt", #The Great Pestilence (A.D. 1348-9), Now Commonly Known as the Black Death
  "https://www.gutenberg.org/cache/epub/44308/pg44308.txt", #The Art of War in the Middle Ages A.D. 378-1515 by Charles Oman
  "https://www.gutenberg.org/cache/epub/42686/pg42686.txt", #A History of Epidemics in Britain, Volume 1 (of 2) by Charles Creighton
  "https://www.gutenberg.org/cache/epub/43671/pg43671.txt", #A History of Epidemics in Britain, Volume 2 (of 2) by Charles Creighton
  "https://www.gutenberg.org/cache/epub/28556/pg28556.txt", #History of Woman Suffrage, Volume III by Stanton, Anthony, and Gage
  "https://www.gutenberg.org/cache/epub/22305/pg22305.txt", #The Naval History of the United States. Volume 1 by Willis J. Abbot
  "https://www.gutenberg.org/cache/epub/26416/pg26416.txt", #The Naval History of the United States. Volume 2 by Willis J. Abbot
  "https://www.gutenberg.org/cache/epub/15735/pg15735.txt", #History of the Negro Race in America from 1619 to 1880. Vol 1 by Williams
  "https://www.gutenberg.org/cache/epub/21851/pg21851.txt", #History of the Negro Race in America from 1619 to 1880. Vol. 2 by Williams
  "https://www.gutenberg.org/cache/epub/21990/pg21990.txt", #The Campaign of 1776 around New York and Brooklyn by Henry Phelps Johnston
  "https://www.gutenberg.org/cache/epub/3072/pg3072.txt",   #Andersonville: A Story of Rebel Military Prisons by John McElroy
  "https://www.gutenberg.org/cache/epub/21012/pg21012.txt", #The loyalists of America and their times : from 1620 to 1816, Vol. 1 of 2
  "https://www.gutenberg.org/cache/epub/24658/pg24658.txt", #The loyalists of America and their times : from 1620 to 1816, Vol. 2 of 2
  "https://www.gutenberg.org/cache/epub/25119/pg25119.txt", #The Conquest of Canada, Vol. 1 by George Warburton
  "https://www.gutenberg.org/cache/epub/34862/pg34862.txt", #The Conquest of Canada, Vol. 2 by George Warburton
  "https://www.gutenberg.org/cache/epub/27430/pg27430.txt", #The Critical Period of American History by John Fiske
  "https://www.gutenberg.org/cache/epub/22131/pg22131.txt", #The Story of the Upper Canadian Rebellion, Volume 1 by John Charles Dent
  "https://www.gutenberg.org/cache/epub/15263/pg15263.txt", #The Underground Railroad by William Still
  "https://www.gutenberg.org/cache/epub/7142/pg7142.txt",   #The History of the Peloponnesian War by Thucydides
  "https://www.gutenberg.org/cache/epub/2707/pg2707.txt",   #The History of Herodotus — Volume 1 by Herodotus
  "https://www.gutenberg.org/cache/epub/2456/pg2456.txt",   #The History of Herodotus — Volume 2 by Herodotus
  "https://www.gutenberg.org/cache/epub/46976/pg46976.txt", #The Anabasis of Alexander : or, The history of the wars and conquests of…
  "https://www.gutenberg.org/cache/epub/2850/pg2850.txt",   #The Wars of the Jews; Or, The History of the Destruction of Jerusalem by Josephus
  "https://www.gutenberg.org/cache/epub/1170/pg1170.txt",   #Anabasis by Xenophon
  "https://www.gutenberg.org/cache/epub/6400/pg6400.txt",   #The Lives of the Twelve Caesars, Complete by Suetonius
  "https://www.gutenberg.org/cache/epub/27114/pg27114.txt", #Ancient States and Empires by John Lord
  "https://www.gutenberg.org/cache/epub/19725/pg19725.txt", #The History of Rome, Books 01 to 08 by Livy
  "https://www.gutenberg.org/cache/epub/10907/pg10907.txt", #The History of Rome, Books 09 to 26 by Livy
  "https://www.gutenberg.org/cache/epub/12582/pg12582.txt", #The History of Rome, Books 27 to 36 by Livy
  "https://www.gutenberg.org/cache/epub/44318/pg44318.txt", #The History of Rome, Books 37 to the End by Livy
  "https://www.gutenberg.org/cache/epub/15784/pg15784.txt", #The Chronology of Ancient Kingdoms Amended by Isaac Newton
  "https://www.gutenberg.org/cache/epub/26390/pg26390.txt", #The Two Great Retreats of History by George Grote and comte de Philippe-Paul Ségur
  "https://www.gutenberg.org/cache/epub/16927/pg16927.txt", #Tacitus: The Histories, Volumes I and II by Cornelius Tacitus
  "https://www.gutenberg.org/cache/epub/57374/pg57374.txt", #Annals and Antiquities of Rajasthan, v. 1 of 3 by James Tod
  "https://www.gutenberg.org/cache/epub/57375/pg57375.txt", #Annals and Antiquities of Rajasthan, v. 2 of 3 by James Tod
  "https://www.gutenberg.org/cache/epub/57376/pg57376.txt", #Annals and Antiquities of Rajasthan, v. 3 of 3 by James Tod
  "https://www.gutenberg.org/cache/epub/2124/pg2124.txt",   #A Record of Buddhistic Kingdoms by Faxian
  "https://www.gutenberg.org/cache/epub/49403/pg49403.txt", #The Seri Indians. (1898 N 17 / 1895-1896 (pages 1-344*)) by W J McGee
  "https://www.gutenberg.org/cache/epub/23224/pg23224.txt", #Historical Introduction to Studies Among the Sedentary Indians of New Mexico;…
  "https://www.gutenberg.org/cache/epub/20218/pg20218.txt", #History of the Incas by Pedro Sarmiento de Gamboa
  "https://www.gutenberg.org/cache/epub/37887/pg37887.txt", #Mexico, Aztec, Spanish and Republican, Vol. 1 of 2 by Brantz Mayer
  "https://www.gutenberg.org/cache/epub/49538/pg49538.txt", #Mexico, Aztec, Spanish and Republican, Vol. 2 of 2 by Brantz Mayer
  "https://www.gutenberg.org/cache/epub/1301/pg1301.txt",   #The French Revolution: A History by Thomas Carlyle
  "https://www.gutenberg.org/cache/epub/17450/pg17450.txt", #The Part Borne by the Dutch in the Discovery of Australia 1606-1765 by J. E. Heeres
  "https://www.gutenberg.org/cache/epub/22762/pg22762.txt", #History of the Rise of the Huguenots, Vol. 1 by Henry Martyn Baird
  "https://www.gutenberg.org/cache/epub/30708/pg30708.txt", #History of the Rise of the Huguenots, Vol. 2 by Henry Martyn Baird
  "https://www.gutenberg.org/cache/epub/16764/pg16764.txt", #History of the Wars, Books I and II: The Persian War by Procopius
  "https://www.gutenberg.org/cache/epub/16765/pg16765.txt", #History of the Wars, Books III and IV: The Vandalic War by Procopius
  "https://www.gutenberg.org/cache/epub/20298/pg20298.txt", #History of the Wars, Books V and VI: The Gothic War by Procopius
  "https://www.gutenberg.org/cache/epub/17037/pg17037.txt", #History of the English People, Volume I by John Richard Green
  "https://www.gutenberg.org/cache/epub/17038/pg17038.txt", #History of the English People, Volume II by John Richard Green
  "https://www.gutenberg.org/cache/epub/20812/pg20812.txt", #History of the English People, Volume III by John Richard Green
  "https://www.gutenberg.org/cache/epub/23317/pg23317.txt", #History of the English People, Volume IV by John Richard Green
  "https://www.gutenberg.org/cache/epub/23642/pg23642.txt", #History of the English People, Volume V by John Richard Green
  "https://www.gutenberg.org/cache/epub/24254/pg24254.txt", #History of the English People, Volume VI by John Richard Green
  "https://www.gutenberg.org/cache/epub/25261/pg25261.txt", #History of the English People, Volume VII by John Richard Green
  "https://www.gutenberg.org/cache/epub/25536/pg25536.txt", #History of the English People, Volume VIII by John Richard Green
  "https://www.gutenberg.org/cache/epub/33504/pg33504.txt", #Opticks : or, A treatise of the reflections, refractions, inflections and…
  "https://www.gutenberg.org/cache/epub/22062/pg22062.txt", #The Mathematicall Praeface to Elements of Geometrie of Euclid of Megara by John Dee
  "https://www.gutenberg.org/cache/epub/26752/pg26752.txt", #The Way To Geometry by Petrus Ramus
  "https://www.gutenberg.org/cache/epub/68662/pg68662.txt", #Elements of arithmetic by Augustus De Morgan
  "https://www.gutenberg.org/cache/epub/5000/pg5000.txt",   #The Notebooks of Leonardo Da Vinci — Complete by da Vinci Leonardo
  "https://www.gutenberg.org/cache/epub/50572/pg50572.txt", #Goethe's Theory of Colours by Johann Wolfgang von Goethe
  "https://www.gutenberg.org/cache/epub/14986/pg14986.txt", #Experimental Researches in Electricity, Volume 1 by Michael Faraday
  "https://www.gutenberg.org/cache/epub/46036/pg46036.txt", #The sidereal messenger of Galileo Galilei : and a part of the preface to…
  "https://www.gutenberg.org/cache/epub/37729/pg37729.txt", #A Discourse Presented to the Most Serene Don Cosimo II., Great Duke of Tuscany,…
  "https://www.gutenberg.org/cache/epub/13476/pg13476.txt", #Experiments with Alternate Currents of High Potential and High Frequency by Tesla
  "https://www.gutenberg.org/cache/epub/33810/pg33810.txt", #On the magnet, magnetick bodies also, and on the great magnet the earth by Gilbert et al.
  "https://www.gutenberg.org/cache/epub/41839/pg41839.txt", #The Boy's Playbook of Science by John Henry Pepper
  "https://www.gutenberg.org/cache/epub/45515/pg45515.txt", #Experiments and Observations on Electricity Made at Philadelphia in America
  "https://www.gutenberg.org/cache/epub/54969/pg54969.txt", #Sound by John Tyndall
  "https://www.gutenberg.org/cache/epub/63803/pg63803.txt", #The Forms of Water in Clouds and Rivers, Ice and Glaciers by John Tyndall
  "https://www.gutenberg.org/cache/epub/52869/pg52869.txt", #On the Connexion of the Physical Sciences by Mary Somerville
  "https://www.gutenberg.org/cache/epub/55886/pg55886.txt", #On Molecular and Microscopic Science, Volume 1 (of 2) by Mary Somerville
  "https://www.gutenberg.org/cache/epub/57566/pg57566.txt", #On Molecular and Microscopic Science, Volume 2 (of 2) by Mary Somerville
  "https://www.gutenberg.org/cache/epub/71191/pg71191.txt", #The celestial worlds discover'd : or, conjectures concerning the inhabitants,…
  "https://www.gutenberg.org/cache/epub/30775/pg30775.txt", #Elements of Chemistry, by Antoine Laurent Lavoisier
  "https://www.gutenberg.org/cache/epub/51326/pg51326.txt", #The Principles of Chemistry, Volume I by Dmitry Ivanovich Mendeleyev
  "https://www.gutenberg.org/cache/epub/54210/pg54210.txt", #The Principles of Chemistry, Volume II by Dmitry Ivanovich Mendeleyev
  "https://www.gutenberg.org/cache/epub/74240/pg74240.txt", #The works of the highly experienced and famous chymist, John Rudolph Glauber…
  "https://www.gutenberg.org/cache/epub/63355/pg63355.txt", #Studies on fermentation : the diseases of beer, their causes, and the means of…
  "https://www.gutenberg.org/cache/epub/29734/pg29734.txt", #Experiments and Observations on Different Kinds of Air by Joseph Priestley
  "https://www.gutenberg.org/cache/epub/37682/pg37682.txt", #Heads of Lectures on a Course of Experimental Philosophy: Particularly…
  "https://www.gutenberg.org/cache/epub/46998/pg46998.txt", #Elements of the Theory and Practice of Chymistry, 5th ed. by Pierre Joseph Macquer
  "https://www.gutenberg.org/cache/epub/58393/pg58393.txt", #The Mirror of Alchimy by Roger Bacon
  "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",   #On the Origin of Species By Means of Natural Selection by Charles Darwin
  "https://www.gutenberg.org/cache/epub/2300/pg2300.txt",   #The Descent of Man, and Selection in Relation to Sex by Charles Darwin
  "https://www.gutenberg.org/cache/epub/24923/pg24923.txt", #The Variation of Animals and Plants Under Domestication, Vol. I. by Charles Darwin
  "https://www.gutenberg.org/cache/epub/1227/pg1227.txt",   #The Expression of the Emotions in Man and Animals by Charles Darwin
  "https://www.gutenberg.org/cache/epub/22728/pg22728.txt", #The Foundations of the Origin of Species by Charles Darwin
  "https://www.gutenberg.org/cache/epub/2355/pg2355.txt",   #The Formation of Vegetable Mould Through the Action of Worms by Charles Darwin
  "https://www.gutenberg.org/cache/epub/28897/pg28897.txt", #The Variation of Animals and Plants Under Domestication, Volume II (of 2) by Darwin
  "https://www.gutenberg.org/cache/epub/2485/pg2485.txt",   #The Movements and Habits of Climbing Plants by Charles Darwin
  "https://www.gutenberg.org/cache/epub/3807/pg3807.txt",   #The Different Forms of Flowers on Plants of the Same Species by Charles Darwin
  "https://www.gutenberg.org/cache/epub/2690/pg2690.txt",   #Coral Reefs by Charles Darwin
  "https://www.gutenberg.org/cache/epub/5765/pg5765.txt",   #Insectivorous Plants by Charles Darwin
  "https://www.gutenberg.org/cache/epub/5605/pg5605.txt",   #The Power of Movement in Plants by Charles Darwin and Sir Francis Darwin
  "https://www.gutenberg.org/cache/epub/3054/pg3054.txt",   #Volcanic Islands by Charles Darwin
  "https://www.gutenberg.org/cache/epub/23354/pg23354.txt", #Vegetable Teratology by Maxwell T. Masters
  "https://www.gutenberg.org/cache/epub/15491/pg15491.txt", #Micrographia by Robert Hooke
  "https://www.gutenberg.org/cache/epub/72894/pg72894.txt", #On the mode of communication of cholera by John Snow
  "https://www.gutenberg.org/cache/epub/68315/pg68315.txt", #On chloroform and other anæsthetics: their action and administration by John Snow
  "https://www.gutenberg.org/cache/epub/22153/pg22153.txt", #Pagan and Christian Rome by Rodolfo Amedeo Lanciani
  "https://www.gutenberg.org/cache/epub/74614/pg74614.txt", #Japan by Walter G. Dickson
  "https://www.gutenberg.org/cache/epub/53360/pg53360.txt", #The History of the Indian Revolt and of the Expeditions to Persia, China and…
  "https://www.gutenberg.org/cache/epub/57628/pg57628.txt", #The Principles of Psychology, Volume 1 (of 2) by William James
  "https://www.gutenberg.org/cache/epub/57634/pg57634.txt", #The Principles of Psychology, Volume 2 (of 2) by William James
  "https://www.gutenberg.org/cache/epub/33224/pg33224.txt", #Principles of Geology by Sir Charles Lyell
  "https://www.gutenberg.org/cache/epub/61652/pg61652.txt", #Physical Geography by Mary Somerville
  "https://www.gutenberg.org/cache/epub/24931/pg24931.txt", #Elements of Agricultural Chemistry by Thomas Anderson
  "https://www.gutenberg.org/cache/epub/47626/pg47626.txt", #The Ocean World: Being a Description of the Sea and Its Living Inhabitants.
  "https://www.gutenberg.org/cache/epub/45873/pg45873.txt", #Reptiles and Birds by Louis Figuier
  "https://www.gutenberg.org/cache/epub/45596/pg45596.txt", #The Insect World by Louis Figuier
  "https://www.gutenberg.org/cache/epub/42380/pg42380.txt", #Primitive Man by Louis Figuier
  "https://www.gutenberg.org/cache/epub/22973/pg22973.txt", #The English Husbandman by Gervase Markham
  "https://www.gutenberg.org/cache/epub/69970/pg69970.txt", #The English Husbandman (The Second Booke) by Gervase Markham

]

# Parsing and corpus preparation.
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|endoftext|>"]
START = re.compile(
    r"^\*\*\*\s*START OF .*?PROJECT GUTENBERG.*?\*\*\*[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
END = re.compile(
    r"^\*\*\*\s*END OF .*?PROJECT GUTENBERG.*?\*\*\*[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_sources(source_values):
    """Validate the embedded URL list; local text paths also work for experiments."""
    if not isinstance(source_values, list) or not source_values:
        raise ValueError("sources must be a nonempty list of URLs or local text paths.")
    entries, seen = [], set()
    for value in source_values:
        if not isinstance(value, str) or not value:
            raise ValueError("Each source must be a nonempty string.")
        parsed = urlparse(value)
        if parsed.scheme not in {"", "http", "https"}:
            raise ValueError(f"Unsupported source: {value}")
        canonical = value if parsed.scheme else str(Path(value).resolve())
        if canonical in seen:
            continue
        seen.add(canonical)
        entries.append({"source": canonical, "title": Path(parsed.path).name})
    return entries


def preparation_signature(args):
    entries = load_sources(args.sources)
    if args.max_sources:
        entries = entries[: args.max_sources]
    payload = {
        "cleaner_version": 1,
        "sources": [entry["source"] for entry in entries],
        "vocab_size": args.vocab_size,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "skip_errors": args.skip_errors,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def download_url(source, mirror):
    parsed = urlparse(source)
    if parsed.hostname in {"gutenberg.org", "www.gutenberg.org"}:
        match = re.fullmatch(r"/cache/epub/(\d+)/pg(\d+)\.txt", parsed.path)
        if match:
            if match[1] != match[2]:
                raise ValueError(f"Mismatched Gutenberg IDs in {source}")
            return f"{mirror.rstrip('/')}/{match[1]}/pg{match[1]}.txt"
    return source


def clean_text(raw, require_wrappers=False):
    raw = unicodedata.normalize(
        "NFC", raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    )
    if raw.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("Expected plain text, received HTML.")
    start, end = START.search(raw), END.search(raw)
    if require_wrappers and (not start or not end):
        raise ValueError(
            "Gutenberg start/end markers are missing; refusing to train on a possible error page."
        )
    if start and end and start.end() >= end.start():
        raise ValueError("Gutenberg markers are out of order.")
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = re.sub(r"\s+", " ", paragraph).strip()
        if text and not re.fullmatch(
            r"\[Illustration[^\]]*\]", text, flags=re.IGNORECASE
        ):
            paragraphs.append(text)
    cleaned = "\n\n".join(paragraphs)
    if len(cleaned) < 200:
        raise ValueError("Fewer than 200 characters remain after cleaning.")
    return cleaned + "\n"


def split_documents(documents, validation_fraction, test_fraction, seed):
    if (
        validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be positive and sum to less than one."
        )
    if len(documents) < 3:
        raise ValueError(
            "At least three distinct books are needed for train/validation/test splits."
        )
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    n_validation = max(1, round(len(shuffled) * validation_fraction))
    n_test = max(1, round(len(shuffled) * test_fraction))
    if n_validation + n_test >= len(shuffled):
        raise ValueError("Split fractions leave no training books.")
    for index, document in enumerate(shuffled):
        document["split"] = (
            "validation"
            if index < n_validation
            else "test"
            if index < n_validation + n_test
            else "train"
        )
    return shuffled


def text_chunks(path, size=16384):
    """Bound tokenizer memory even for multi-volume reference books."""
    with Path(path).open(encoding="utf-8") as stream:
        while chunk := stream.read(size):
            yield chunk


def prepare(args):

    if not 259 <= args.vocab_size <= 65535:
        raise ValueError(
            "vocab-size must be between 259 and 65535 for byte BPE and uint16 storage."
        )
    if args.download_delay < 0 or args.max_sources < 0:
        raise ValueError("Download delay and max-sources must be nonnegative.")
    if (
        args.validation_fraction <= 0
        or args.test_fraction <= 0
        or args.validation_fraction + args.test_fraction >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be positive and sum to less than one."
        )
    output, cache = Path(args.data_dir), Path(args.cache_dir)
    if output.exists():
        raise ValueError(
            f"{output} already exists. Use a new CONFIG['data_dir'] to preserve existing splits."
        )
    entries = load_sources(args.sources)
    if args.max_sources:
        entries = entries[: args.max_sources]
    output.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "tidbitmodel/0.1 (corpus preparation)"
    try:
        with tempfile.TemporaryDirectory(
            prefix=".prepare-", dir=output.parent
        ) as temporary:
            staging = Path(temporary)
            (staging / "clean").mkdir()
            documents, failures, duplicates, hashes = [], [], [], set()
            for index, entry in enumerate(entries, 1):
                source = entry["source"]
                try:
                    is_remote = urlparse(source).scheme in {"http", "https"}
                    is_gutenberg = urlparse(source).hostname in {
                        "gutenberg.org",
                        "www.gutenberg.org",
                    }
                    if is_remote:
                        url = download_url(source, args.mirror)
                        raw_path = cache / (
                            hashlib.sha256(source.encode()).hexdigest() + ".txt"
                        )
                        if not raw_path.exists():
                            part = raw_path.with_suffix(".part")
                            try:
                                with session.get(
                                    url, timeout=(10, 60), stream=True
                                ) as response:
                                    response.raise_for_status()
                                    with part.open("wb") as stream:
                                        for chunk in response.iter_content(
                                            chunk_size=65536
                                        ):
                                            stream.write(chunk)
                                # Validate downloaded content before making it a reusable cache entry.
                                clean_text(
                                    part.read_text(encoding="utf-8-sig"),
                                    require_wrappers=is_gutenberg,
                                )
                                part.replace(raw_path)
                            finally:
                                part.unlink(missing_ok=True)
                                time.sleep(args.download_delay)
                    else:
                        raw_path = Path(source)
                    raw = raw_path.read_text(encoding="utf-8-sig")
                    title = re.search(r"^Title:[ \t]*(.+)$", raw[:20000], re.MULTILINE)
                    if title:
                        entry["title"] = title.group(1).strip()
                    text = clean_text(raw, require_wrappers=is_remote and is_gutenberg)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest in hashes:
                        duplicates.append(source)
                        continue
                    hashes.add(digest)
                    relative = f"clean/{digest}.txt"
                    (staging / relative).write_text(text, encoding="utf-8")
                    documents.append(
                        {
                            **entry,
                            "sha256": digest,
                            "path": relative,
                            "characters": len(text),
                        }
                    )
                    print(
                        f"[{index}/{len(entries)}] Parsed {entry['title']} ({len(text):,} characters)",
                        flush=True,
                    )
                except (
                    OSError,
                    UnicodeError,
                    ValueError,
                    requests.RequestException,
                ) as exc:
                    failures.append({"source": source, "error": str(exc)})
                    print(
                        f"[{index}/{len(entries)}] FAILED {source}: {exc}", flush=True
                    )
            write_json(
                cache / "last-prepare-report.json",
                {"failures": failures, "duplicate_sources": duplicates},
            )
            if failures and not args.skip_errors:
                raise ValueError(
                    f"{len(failures)} sources failed; see {cache / 'last-prepare-report.json'}. Fix them or set CONFIG['skip_errors'] = True."
                )
            documents = split_documents(
                documents, args.validation_fraction, args.test_fraction, args.seed
            )
            backend = Tokenizer(models.BPE())
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=args.vocab_size,
                min_frequency=2,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=SPECIAL_TOKENS,
            )
            training_text = (
                chunk
                for document in documents
                if document["split"] == "train"
                for chunk in text_chunks(staging / document["path"])
            )
            print(
                "Training a byte-level BPE tokenizer on training books only...",
                flush=True,
            )
            backend.train_from_iterator(training_text, trainer=trainer)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token=SPECIAL_TOKENS[0],
                bos_token=SPECIAL_TOKENS[1],
                eos_token=SPECIAL_TOKENS[2],
            )
            tokenizer.save_pretrained(staging / "tokenizer")
            counts = {}
            for split in ("train", "validation", "test"):
                count = 0
                with (staging / f"{split}.bin").open("wb") as stream:
                    for document in documents:
                        if document["split"] != split:
                            continue
                        before = count
                        np.asarray([tokenizer.bos_token_id], dtype="<u2").tofile(stream)
                        count += 1
                        for chunk in text_chunks(staging / document["path"]):
                            ids = backend.encode(chunk, add_special_tokens=False).ids
                            np.asarray(ids, dtype="<u2").tofile(stream)
                            count += len(ids)
                        np.asarray([tokenizer.eos_token_id], dtype="<u2").tofile(stream)
                        count += 1
                        document["tokens"] = count - before
                counts[split] = count
            fingerprint = hashlib.sha256(
                (json.dumps(documents, sort_keys=True) + backend.to_str()).encode()
            ).hexdigest()
            metadata = {
                "format_version": 1,
                "preparation_signature": preparation_signature(args),
                "dtype": "<u2",
                "seed": args.seed,
                "vocab_size": len(tokenizer),
                "token_counts": counts,
                "fingerprint": fingerprint,
                "documents": documents,
                "failed_sources": failures,
                "duplicate_sources": duplicates,
                "split_unit": "whole book; exact duplicate books removed before splitting",
            }
            write_json(staging / "metadata.json", metadata)
            # Publish only a complete dataset. Downloads remain reusable after failures.
            shutil.move(str(staging), str(output))
    finally:
        session.close()
    print(
        json.dumps(
            {
                "data_dir": str(output),
                "books": len(documents),
                "vocab_size": len(tokenizer),
                "tokens": counts,
            },
            indent=2,
        )
    )


class TokenBlocks:
    """Memory-mapped causal-LM sequences with each target counted once.

    Adjacent blocks overlap by one context token. The final short block is
    padded and its padding labels are masked. EOS/BOS delimit packed books.
    """

    def __init__(self, data_dir, split, context, pad_token_id):
        self.torch = torch
        self.context, self.pad = context, pad_token_id
        metadata = json.loads((Path(data_dir) / "metadata.json").read_text())
        path = Path(data_dir) / f"{split}.bin"
        dtype = np.dtype(metadata["dtype"])
        if path.stat().st_size != metadata["token_counts"][split] * dtype.itemsize:
            raise ValueError(f"{path} does not match its recorded token count.")
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        if context < 2 or len(self.tokens) < 2:
            raise ValueError(
                "Context and token stream must each contain at least two tokens."
            )

    def __len__(self):
        return (len(self.tokens) - 2) // (self.context - 1) + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * (self.context - 1)
        chunk = np.asarray(self.tokens[start : start + self.context], dtype=np.int64)
        size = len(chunk)
        inputs = self.torch.full((self.context,), self.pad, dtype=self.torch.long)
        inputs[:size] = self.torch.from_numpy(chunk.copy())
        labels = inputs.clone()
        labels[size:] = -100
        mask = self.torch.zeros(self.context, dtype=self.torch.long)
        mask[:size] = 1
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


# Model training, checkpointing, and evaluation.
def training_configuration(args):
    return {
        "context": args.context,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
    }


SAMPLE_EVERY = 100


def choose_device(requested):
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def read_prompts(prompts):
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        raise ValueError("sample_prompts must be a nonempty list of strings.")
    return list(prompts)


def sample_responses(
    model, tokenizer, prompts, device, max_new_tokens=64, temperature=0.8, seed=42
):
    """Generate without consuming training RNG state or leaving dropout disabled."""
    if max_new_tokens < 1 or temperature <= 0:
        raise ValueError("max-new-tokens and temperature must be positive.")
    previous_mode, previous_rng = model.training, rng_state(device)
    samples = []
    try:
        torch.manual_seed(seed)
        model.eval()
        budget = min(max_new_tokens, model.config.n_positions - 1)
        with torch.inference_mode():
            for prompt in prompts:
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                ids = (ids or [tokenizer.bos_token_id])[
                    -(model.config.n_positions - budget) :
                ]
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=budget,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=min(50, len(tokenizer)),
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=[[tokenizer.pad_token_id], [tokenizer.bos_token_id]],
                    use_cache=True,
                )
                continuation = tokenizer.decode(
                    output[0, len(ids) :], skip_special_tokens=True
                )
                samples.append({"prompt": prompt, "continuation": continuation})
    finally:
        model.train(previous_mode)
        restore_rng(previous_rng, device)
    return samples


def show_samples(samples, step=None):
    label = f" at step {step}" if step is not None else ""
    print(f"\n--- Sample responses{label} ---", flush=True)
    for sample in samples:
        print(
            f"Prompt: {sample['prompt']}\nResponse: {sample['continuation']}\n",
            flush=True,
        )


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def evaluate(model, dataset, device, batch_size, max_batches=0):
    """Token-weighted loss; padding and the first context token are not targets."""
    previous_mode = model.training
    total_loss, total_targets, batches = 0.0, 0, 0
    # A dedicated generator prevents DataLoader setup from consuming training RNG.
    limit = max_batches * batch_size
    selection = dataset
    if limit and limit < len(dataset):
        # Cover the split instead of evaluating only the opening of its first book.
        selection = Subset(
            dataset, np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
        )
    loader = DataLoader(
        selection,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                targets = int((batch["labels"][:, 1:] != -100).sum().item())
                loss = model(**batch, use_cache=False).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Non-finite evaluation loss.")
                total_loss += loss * targets
                total_targets += targets
                batches += 1
                if max_batches and batches >= max_batches:
                    break
    finally:
        model.train(previous_mode)
    if total_targets == 0:
        raise ValueError("No evaluation targets were found.")
    mean = total_loss / total_targets
    return {
        "loss": mean,
        "perplexity": math.exp(mean) if mean < 700 else None,
        "target_tokens": total_targets,
        "batches": batches,
        "complete_split": len(selection) == len(dataset),
    }


def build_model(tokenizer, args):
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads.")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.context,
        n_ctx=args.context,
        n_embd=args.width,
        n_layer=args.layers,
        n_head=args.heads,
        n_inner=4 * args.width,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # This constructor initializes random weights. No pretrained GPT-2 weights.
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config)


def save_checkpoint(
    path, model, tokenizer, optimizer, scheduler, sampler, step, best_loss, run, device
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "run.json", run)
    temporary = path / "training_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "best_validation_loss": best_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler_rng": sampler.get_state(),
            "rng": rng_state(device),
        },
        temporary,
    )
    temporary.replace(path / "training_state.pt")


def train(args):
    if (
        min(
            args.max_steps,
            args.batch_size,
            args.gradient_accumulation,
            args.layers,
            args.width,
            args.heads,
            args.threads,
            args.eval_every,
            args.save_every,
            args.log_every,
        )
        <= 0
    ):
        raise ValueError(
            "Step intervals, batch sizes, model dimensions, and threads must be positive."
        )
    if (
        args.context < 2
        or not 0 <= args.dropout < 1
        or args.warmup_steps < 0
        or args.eval_batches < 0
    ):
        raise ValueError(
            "Invalid context, dropout, warmup, or evaluation-batch setting."
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if args.sample_max_new_tokens < 1:
        raise ValueError("sample-max-new-tokens must be positive.")
    prompts = read_prompts(args.sample_prompts)
    output, data = Path(args.output_dir), Path(args.data_dir)
    metadata = json.loads((data / "metadata.json").read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        data / "tokenizer", local_files_only=True
    )
    device = choose_device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    configuration = training_configuration(args)
    run = {
        "dataset_fingerprint": metadata["fingerprint"],
        "configuration": configuration,
        "device": str(device),
        "sample_every": SAMPLE_EVERY,
        "sample_prompts": prompts,
        "sample_max_new_tokens": args.sample_max_new_tokens,
        "torch_version": str(torch.__version__),
    }
    state = None
    if args.resume:
        checkpoint = Path(args.resume)
        previous = json.loads((checkpoint / "run.json").read_text())
        if previous["dataset_fingerprint"] != run["dataset_fingerprint"]:
            raise ValueError("Checkpoint and prepared corpus/tokenizer do not match.")
        if previous["configuration"] != configuration:
            raise ValueError(
                "Resume requires the original training settings, including the planned max-steps."
            )
        model = GPT2LMHeadModel.from_pretrained(
            checkpoint, local_files_only=True, attn_implementation="eager"
        )
        state = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"{output} is not empty. Use a new output directory or CONFIG['resume']."
            )
        model = build_model(tokenizer, args)
    if output.exists() and any(output.iterdir()):
        existing = output / "run.json"
        existing_run = json.loads(existing.read_text()) if existing.exists() else {}
        if (
            existing_run.get("dataset_fingerprint") != metadata["fingerprint"]
            or existing_run.get("configuration") != configuration
        ):
            raise ValueError("The output directory belongs to a different run.")
    output.mkdir(parents=True, exist_ok=True)
    model.to(device)
    run["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    write_json(output / "run.json", run)
    training = TokenBlocks(data, "train", args.context, tokenizer.pad_token_id)
    validation = TokenBlocks(data, "validation", args.context, tokenizer.pad_token_id)
    groups = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    warmup = min(args.warmup_steps, max(0, args.max_steps - 1))

    def learning_rate_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, args.max_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    start, best_loss = 0, float("inf")
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.set_state(state["sampler_rng"])
        restore_rng(state["rng"], device)
        start, best_loss = state["step"], state["best_validation_loss"]
        # A resumed run in a new directory needs its own best checkpoint.
        # The previous best model may belong to an earlier, unavailable step.
        if not (output / "best" / "training_state.pt").exists():
            best_loss = float("inf")
    if start >= args.max_steps:
        raise ValueError("This checkpoint already reached the planned max-steps.")
    print(
        f"Training {run['parameters']:,} randomly initialized parameters on {device}; samples every {SAMPLE_EVERY} optimizer steps.",
        flush=True,
    )
    print(
        f"Train tokens: {len(training.tokens):,}; validation tokens: {len(validation.tokens):,}",
        flush=True,
    )
    model.train()
    for step in range(start + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for _ in range(args.gradient_accumulation):
            indices = torch.randint(
                len(training), (args.batch_size,), generator=sampler
            ).tolist()
            rows = [training[index] for index in indices]
            batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
            targets = int((batch["labels"][:, 1:] != -100).sum().item())
            microbatches.append((batch, targets))
        total_targets = sum(targets for _, targets in microbatches)
        train_loss = 0.0
        for batch, targets in microbatches:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at step {step}.")
            weight = targets / total_targets
            (loss * weight).backward()
            train_loss += loss.detach().item() * weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            metrics = {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            print(json.dumps(metrics), flush=True)
            append_json(output / "metrics.jsonl", metrics)
        if step % SAMPLE_EVERY == 0:
            samples = sample_responses(
                model,
                tokenizer,
                prompts,
                device,
                args.sample_max_new_tokens,
                seed=args.seed,
            )
            show_samples(samples, step)
            append_json(output / "samples.jsonl", {"step": step, "samples": samples})
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(
                model, validation, device, args.batch_size, args.eval_batches
            )
            print(f"Validation at step {step}: {json.dumps(metrics)}", flush=True)
            append_json(output / "metrics.jsonl", {"step": step, "validation": metrics})
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(
                    output / "best",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    sampler,
                    step,
                    best_loss,
                    run,
                    device,
                )
        if step % args.save_every == 0:
            save_checkpoint(
                output / f"checkpoint-{step}",
                model,
                tokenizer,
                optimizer,
                scheduler,
                sampler,
                step,
                best_loss,
                run,
                device,
            )
    save_checkpoint(
        output / "model",
        model,
        tokenizer,
        optimizer,
        scheduler,
        sampler,
        step,
        best_loss,
        run,
        device,
    )
    print(
        f"Saved final model to {output / 'model'}; best validation checkpoint to {output / 'best'}.",
        flush=True,
    )


def load_model(path, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        path, local_files_only=True, attn_implementation="eager"
    )
    return model.to(device), tokenizer


def test_model(args):
    if args.batch_size < 1 or args.max_batches < 0 or args.threads < 1:
        raise ValueError(
            "Batch size and threads must be positive; max-batches must be nonnegative."
        )
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model_dir, device)
    metadata = json.loads((Path(args.data_dir) / "metadata.json").read_text())
    run = json.loads((Path(args.model_dir) / "run.json").read_text())
    if metadata["fingerprint"] != run["dataset_fingerprint"]:
        raise ValueError(
            "The evaluation corpus/tokenizer differs from the training dataset."
        )
    dataset = TokenBlocks(
        args.data_dir, args.split, model.config.n_positions, tokenizer.pad_token_id
    )
    metrics = evaluate(model, dataset, device, args.batch_size, args.max_batches)
    samples = sample_responses(
        model,
        tokenizer,
        read_prompts(args.sample_prompts),
        device,
        args.max_new_tokens,
        seed=args.seed,
    )
    result = {
        "split": args.split,
        **metrics,
        "samples": samples,
        "note": "Perplexity measures language modeling, not factual correctness.",
    }
    destination = (
        Path(args.report)
        if args.report
        else Path(args.model_dir).parent / f"{args.split}-results.json"
    )
    write_json(destination, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples"}, indent=2
        )
    )
    show_samples(samples)
    print(f"Saved evaluation report to {destination}")


def interactive_test(args, input_fn=None):
    """Let the user enter prompts repeatedly; also works with notebook input()."""
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    if args.max_new_tokens < 1 or args.temperature <= 0:
        raise ValueError("max_new_tokens and temperature must be positive.")
    input_fn = input if input_fn is None else input_fn
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model_dir, device)
    print(
        "\nInteractive testing is ready. Enter a prompt; type /quit to finish.",
        flush=True,
    )
    number = 0
    while True:
        try:
            prompt = input_fn("\nYour prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive testing finished.", flush=True)
            break
        if prompt.lower() in {"/quit", "/exit"}:
            print("Interactive testing finished.", flush=True)
            break
        if not prompt:
            continue
        samples = sample_responses(
            model,
            tokenizer,
            [prompt],
            device,
            args.max_new_tokens,
            args.temperature,
            args.seed + number,
        )
        print("Response: " + samples[0]["continuation"], flush=True)
        number += 1


def validate_saved_run(checkpoint, args, metadata):
    saved = json.loads((Path(checkpoint) / "run.json").read_text())
    if saved["dataset_fingerprint"] != metadata["fingerprint"]:
        raise ValueError(
            "The saved run uses a different prepared corpus/tokenizer. Choose a new output_dir."
        )
    if saved["configuration"] != training_configuration(args):
        raise ValueError(
            "The saved run uses different training settings. Restore those settings or choose a new output_dir."
        )


def run(config=None, source_list=None, input_fn=None):
    """Execute selected stages; importing this file alone does not start a run."""
    settings = dict(CONFIG)
    if config is not None:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f"Unknown CONFIG keys: {sorted(unknown)}")
        settings.update(config)
    args = SimpleNamespace(**settings)
    args.sources = list(sources if source_list is None else source_list)
    if args.max_sources < 0:
        raise ValueError("max_sources must be nonnegative.")
    data, output = Path(args.data_dir), Path(args.output_dir)
    fallback_model = output / "model"
    if args.prepare:
        if data.exists():
            metadata = json.loads((data / "metadata.json").read_text())
            if metadata.get("preparation_signature") != preparation_signature(args):
                raise ValueError(
                    "Prepared data uses different sources/settings. Choose a new CONFIG['data_dir']."
                )
            if metadata.get("failed_sources"):
                print(
                    f"Reusing a partial corpus with {len(metadata['failed_sources'])} recorded source failures.",
                    flush=True,
                )
            print(f"Reusing prepared corpus at {data}", flush=True)
        else:
            prepare(args)
    if args.train:
        metadata = json.loads((data / "metadata.json").read_text())
        finished = output / "model"
        completed = False
        if not args.resume and (finished / "training_state.pt").is_file():
            validate_saved_run(finished, args, metadata)
            state = torch.load(
                finished / "training_state.pt", map_location="cpu", weights_only=True
            )
            completed = state["step"] >= args.max_steps
            if completed:
                print(
                    f"Reusing completed training run at {finished} (step {state['step']}).",
                    flush=True,
                )
            else:
                args.resume = finished
            del state
        if not completed:
            if not args.resume and output.exists():
                candidates = list(output.glob("checkpoint-*/training_state.pt"))
                candidates += list(output.glob("best/training_state.pt"))
                latest_step, latest = -1, None
                for candidate in candidates:
                    state = torch.load(candidate, map_location="cpu", weights_only=True)
                    if state["step"] > latest_step:
                        latest_step, latest = state["step"], candidate.parent
                    del state
                if latest:
                    args.resume = latest
            if args.resume:
                validate_saved_run(args.resume, args, metadata)
                state = torch.load(
                    Path(args.resume) / "training_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                completed = state["step"] >= args.max_steps
                del state
                if completed:
                    fallback_model = Path(args.resume)
                    print(f"Reusing completed checkpoint at {args.resume}", flush=True)
                else:
                    print(f"Resuming training from {args.resume}", flush=True)
            if not completed:
                train(args)
    if not args.model_dir:
        args.model_dir = (
            output / "best"
            if (output / "best" / "config.json").is_file()
            else fallback_model
        )
    if args.evaluate:
        print(
            f"\nEvaluating the {args.split} split using {args.model_dir}...", flush=True
        )
        test_model(args)
    if args.interactive:
        interactive_test(args, input_fn=input_fn)
    return (
        Path(args.model_dir)
        if args.train or args.evaluate or args.interactive
        else None
    )


if __name__ == "__main__":
    run()
