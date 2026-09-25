import re
import unittest

from ir_datasets.datasets.core import CoreDocument
from ir_datasets.formats import GenericDoc, GenericQuery, TrecQrel

from .base import DatasetIntegrationTest


class TestCore(DatasetIntegrationTest):
    def test_passage_docs(self):
        self._test_docs('core/passage/core', count=7126, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7125: GenericDoc('msmarco_passage_07_688498442', re.compile('^Secondly, the track spike has a curved and tapered toe\\. The amount of tapper differs slightly based .{83} taper\\. Finally the track spike has short spikes located only on the balls of the feet and the toes\\.$', flags=48)),
        })
        self._test_docs('core/passage/10k', count=10000, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7126: GenericDoc('msmarco_passage_14_862119130', re.compile("^Chip's Fab Cobb Salad\\. \\$9\\.99\\. Grilled Chicken, Crispy Bacon, Tomatoes, Egg \\& Crumbled Bleu Cheese on.{107}Mixed Greens\\. Del Rio Taco Salad\\. \\$9\\.99\\. Taco Meat, Tomatoes, Cheddar Cheese, Chives \\& 'Tilla Chips\\.$", flags=48)),
            9999: GenericDoc('msmarco_passage_28_824152286', re.compile('^Margaret Howe in the new documentary “The Girl Who Talked to Dolphins” BBC\\. Peter may not have been .{107}ng being together, and wanting to be together, and missing him when he wasn’t there,” Howe admitted\\.$', flags=48)),
        })
        self._test_docs('core/passage/100k', count=100000, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7126: GenericDoc('msmarco_passage_14_862119130', re.compile("^Chip's Fab Cobb Salad\\. \\$9\\.99\\. Grilled Chicken, Crispy Bacon, Tomatoes, Egg \\& Crumbled Bleu Cheese on.{107}Mixed Greens\\. Del Rio Taco Salad\\. \\$9\\.99\\. Taco Meat, Tomatoes, Cheddar Cheese, Chives \\& 'Tilla Chips\\.$", flags=48)),
            10000: GenericDoc('msmarco_passage_44_171903413', re.compile('^Underbarrel\\. Go with the Ranger Foregrip here\\. It increases your recoil control and aiming stability.{132}cy and damage range\\. An alternative is the Commando Foregrip, which just has a slightly lower range\\.$', flags=48)),
            99999: GenericDoc('msmarco_passage_47_106711988', re.compile('^The DOL considers the following activities included in management duties: Interviewing, selecting, a.{90}ng the work of employees\\. Maintaining production or sales records for use in supervision or control\\.$', flags=48)),
        })
        self._test_docs('core/passage/1M', count=1000000, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7126: GenericDoc('msmarco_passage_14_862119130', re.compile("^Chip's Fab Cobb Salad\\. \\$9\\.99\\. Grilled Chicken, Crispy Bacon, Tomatoes, Egg \\& Crumbled Bleu Cheese on.{107}Mixed Greens\\. Del Rio Taco Salad\\. \\$9\\.99\\. Taco Meat, Tomatoes, Cheddar Cheese, Chives \\& 'Tilla Chips\\.$", flags=48)),
            10000: GenericDoc('msmarco_passage_44_171903413', re.compile('^Underbarrel\\. Go with the Ranger Foregrip here\\. It increases your recoil control and aiming stability.{132}cy and damage range\\. An alternative is the Commando Foregrip, which just has a slightly lower range\\.$', flags=48)),
            100000: GenericDoc('msmarco_passage_46_561288332', re.compile('^1 cup plain yogurt\\. 2 tablespoons minced fresh ginger\\. 2 tablespoons lemon juice\\. 1 teaspoon kosher .{64}ts\\. Tikka Masala Sauce, recipe follows\\. Tikka Masala Sauce: 1 stick \\(8 tablespoons\\) unsalted butter\\.$', flags=48)),
            999999: GenericDoc('msmarco_passage_00_484900930', re.compile('^25000 km/ 12 months\\. 0W\\-40, 5W\\-40\\. C 230 Kompressor 203\\.040/240/740\\. 5,5 L, Filter capacity 0,4 L\\. 5.{7}W\\-30, 0W\\-40, 0W\\-30\\. C 230 Kompressor 203\\.040/240/740\\. 5,5\\-8 L\\. 13000 miles/ 12 months\\. 0W\\-40, 5W\\-40\\.$', flags=48)),
        }, test_docstore=False, test_iter_split=False)
        self._test_docs('core/passage/10M', count=10000000, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7126: GenericDoc('msmarco_passage_14_862119130', re.compile("^Chip's Fab Cobb Salad\\. \\$9\\.99\\. Grilled Chicken, Crispy Bacon, Tomatoes, Egg \\& Crumbled Bleu Cheese on.{107}Mixed Greens\\. Del Rio Taco Salad\\. \\$9\\.99\\. Taco Meat, Tomatoes, Cheddar Cheese, Chives \\& 'Tilla Chips\\.$", flags=48)),
            10000: GenericDoc('msmarco_passage_44_171903413', re.compile('^Underbarrel\\. Go with the Ranger Foregrip here\\. It increases your recoil control and aiming stability.{132}cy and damage range\\. An alternative is the Commando Foregrip, which just has a slightly lower range\\.$', flags=48)),
            100000: GenericDoc('msmarco_passage_46_561288332', re.compile('^1 cup plain yogurt\\. 2 tablespoons minced fresh ginger\\. 2 tablespoons lemon juice\\. 1 teaspoon kosher .{64}ts\\. Tikka Masala Sauce, recipe follows\\. Tikka Masala Sauce: 1 stick \\(8 tablespoons\\) unsalted butter\\.$', flags=48)),
            1000000: GenericDoc('msmarco_passage_61_661388073', re.compile('^The Establishment of the English colonies in America\\. Topics: Massachusetts, Puritan, Thirteen Colon.{215}e to expand the British Empire, establishing order, protecting colonies and to rehabilitate debtors\\.$', flags=48)),
            9999999: GenericDoc('msmarco_passage_34_781170373', re.compile('^Sony DVP\\-NC685V 5\\-Disc DVD/CD/SACD Player\\. Features: Progressive\\-scan DVD and SACD player, 5\\-disc ca.{106}\\), composite/RCA \\(x2\\), optical, and coaxial outputs\\. Compatible with Dolby Digital and DTS decoding\\.$', flags=48)),
        }, test_docstore=False, test_iter_split=False)
        self._test_docs('core/passage', count=100000000, items={
            0: GenericDoc('msmarco_passage_04_22754188', re.compile('^Support began to grow for the idea of a post\\-war military alliance with the countries of western Eur.{125}ual defense agreement involving the United States and Canada was necessary due to the Soviet threat\\.$', flags=48)),
            9: GenericDoc('msmarco_passage_26_484957863', re.compile('^If you like the look, you can skip the next step and go directly to adding Triple Thick\\. I wanted my.{66}ayer of Mod Podge over the glitter and repeat the same step of pouring the glitter on the mod podge\\.$', flags=48)),
            7126: GenericDoc('msmarco_passage_14_862119130', re.compile("^Chip's Fab Cobb Salad\\. \\$9\\.99\\. Grilled Chicken, Crispy Bacon, Tomatoes, Egg \\& Crumbled Bleu Cheese on.{107}Mixed Greens\\. Del Rio Taco Salad\\. \\$9\\.99\\. Taco Meat, Tomatoes, Cheddar Cheese, Chives \\& 'Tilla Chips\\.$", flags=48)),
            10000: GenericDoc('msmarco_passage_44_171903413', re.compile('^Underbarrel\\. Go with the Ranger Foregrip here\\. It increases your recoil control and aiming stability.{132}cy and damage range\\. An alternative is the Commando Foregrip, which just has a slightly lower range\\.$', flags=48)),
            100000: GenericDoc('msmarco_passage_46_561288332', re.compile('^1 cup plain yogurt\\. 2 tablespoons minced fresh ginger\\. 2 tablespoons lemon juice\\. 1 teaspoon kosher .{64}ts\\. Tikka Masala Sauce, recipe follows\\. Tikka Masala Sauce: 1 stick \\(8 tablespoons\\) unsalted butter\\.$', flags=48)),
            1000000: GenericDoc('msmarco_passage_61_661388073', re.compile('^The Establishment of the English colonies in America\\. Topics: Massachusetts, Puritan, Thirteen Colon.{215}e to expand the British Empire, establishing order, protecting colonies and to rehabilitate debtors\\.$', flags=48)),
            10000000: GenericDoc('msmarco_passage_39_551460284', re.compile('^The second rule of thumb for business valuation is to establish the asset value of the business\\. Fir.{62}ing inventory of all the physical aspects of the business such as fixtures, equipment and inventory\\.$', flags=48)),
            99999999: GenericDoc('msmarco_passage_64_707625947', re.compile('^Sightseeing Tours in New York Factory Tours in New York Bus Tours in New York Segway Tours in New Yo.{244}w York Waterskiing \\& Jetskiing in New York Duck Tours in New York Beer Tastings \\& Tours in New York\\.$', flags=48)),
        }, test_docstore=False, test_iter_split=False)

    def test_passage_queries(self):
        self._test_queries('core/passage', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/passage/core', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/passage/10k', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/passage/100k', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/passage/1M', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/passage/10M', count=65, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2004369', 'how too make homade buttermilk'),
            64: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })

    def test_passage_qrels(self):
        self._test_qrels('core/passage', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })
        self._test_qrels('core/passage/core', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })
        self._test_qrels('core/passage/10k', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })
        self._test_qrels('core/passage/100k', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })
        self._test_qrels('core/passage/1M', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })
        self._test_qrels('core/passage/10M', count=7150, items={
            0: TrecQrel('2001010', 'msmarco_passage_06_483599029', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_passage_47_93817570', 1, '0'),
            7149: TrecQrel('3100918', 'msmarco_passage_66_43181783', 0, '0'),
        })

    def test_document_docs(self):
        self._test_docs('core/document/core', count=6032, items={
            0: CoreDocument('msmarco_doc_29_356827432', 'Allow changes to parts of a protected document - Word', re.compile('^Allow changes to parts of a protected document\nAllow changes to parts of a protected document\nProtec.{108} document\nFind parts you can edit\nThis modification is not allowed because the selection is locked\\.\n$', flags=48), re.compile('^Allow changes to parts of a protected document \\- Word\nAllow changes to parts of a protected document.{14374} document, workbook, or presentation\nA subscription to make the most of your time\nTry one month free$', flags=48)),
            9: CoreDocument('msmarco_doc_07_1025413044', 'How to Slow Cook Beef Top Round Roasts | Livestrong.com', re.compile("^How to Slow Cook Beef Top Round Roasts\nHow to Slow Cook Beef Top Round Roasts\nTip\nCrockpot Top Round.{4}st Recipe\nThings You'll Need\nFrequently Asked Questions on Roasts\nMeat Eating Guidelines\nreferences\n$", flags=48), re.compile('^How to Slow Cook Beef Top Round Roasts \\| Livestrong\\.com\nNutrition\nCooking and Baking\nCooking Techniq.{4782}uide to the Leanest Selections"\nU\\.S\\. Department of Agriculture: "10 Tips: Vary Your Protein Routine"$', flags=48)),
            6031: CoreDocument('msmarco_doc_08_1483055463', 'German Dinner Menu Ideas made Just like Oma', re.compile("^Dinner Menu Ideas\nDinner Menu Ideas\nQuick Dinner Ideas\nGreen Bean Soup Dinner\nBreaded Pork Chop Dinn.{287}ke Rouladen made Just like Oma\nOma's German Asparagus Salad Recipe \\~ Spargelsalat\nWords to the Wise\n$", flags=48), re.compile('^German Dinner Menu Ideas made Just like Oma\nHome\nDinners\nDinner Menu Ideas\n➤ by Oma Gerhild Fulson\nT.{4986}ose who posses it, but discipline is wasted on fools\\."\nProverbs 16:22 \\(NLT\\)\nTop of Dinner Menu Ideas$', flags=48)),
        })
        self._test_docs('core/document/10k', count=10000, items={
            0: CoreDocument('msmarco_doc_29_356827432', 'Allow changes to parts of a protected document - Word', re.compile('^Allow changes to parts of a protected document\nAllow changes to parts of a protected document\nProtec.{108} document\nFind parts you can edit\nThis modification is not allowed because the selection is locked\\.\n$', flags=48), re.compile('^Allow changes to parts of a protected document \\- Word\nAllow changes to parts of a protected document.{14374} document, workbook, or presentation\nA subscription to make the most of your time\nTry one month free$', flags=48)),
            9: CoreDocument('msmarco_doc_07_1025413044', 'How to Slow Cook Beef Top Round Roasts | Livestrong.com', re.compile("^How to Slow Cook Beef Top Round Roasts\nHow to Slow Cook Beef Top Round Roasts\nTip\nCrockpot Top Round.{4}st Recipe\nThings You'll Need\nFrequently Asked Questions on Roasts\nMeat Eating Guidelines\nreferences\n$", flags=48), re.compile('^How to Slow Cook Beef Top Round Roasts \\| Livestrong\\.com\nNutrition\nCooking and Baking\nCooking Techniq.{4782}uide to the Leanest Selections"\nU\\.S\\. Department of Agriculture: "10 Tips: Vary Your Protein Routine"$', flags=48)),
            6032: CoreDocument('msmarco_doc_13_766206358', 'which group of players in the game of economics both produces and uses goods and services - Brainly.com', re.compile('^Which group of players in the game of economics both produces and uses goods and services\nWhich grou.{210}estion\\?\nStill have questions\\?\nNew questions in Social Studies\nGet more out of Brainly\nJoin for free\n$', flags=48), re.compile('^which group of players in the game of economics both produces and uses goods and services \\- Brainly\\..{4016}our own questions\ncheck\nLearn with a massive community\nJoin for free\nAlready have an account\\? Log in$', flags=48)),
            9999: CoreDocument('msmarco_doc_36_1650893521', 'Army COOL Summary - MOS 79V - Army Reserve Career Counselor (Army Reserve)', 'MOS 79V Army Reserve Career Counselor (Army Reserve)\nMOS 79V Army Reserve Career Counselor (Army Reserve)\nMOS Description\n', re.compile('^Army COOL Summary \\- MOS 79V \\- Army Reserve Career Counselor \\(Army Reserve\\)\n\n\n\n\n\n\n\nMOS 79V Army Reser.{18128} education programs when using tuition assistance, see:\nArmyIgnitED\nTuition Assistance \\(TA\\) DECIDE\n\n$', flags=48)),
        })
        self._test_docs('core/document/100k', count=100000, items={
            0: CoreDocument('msmarco_doc_29_356827432', 'Allow changes to parts of a protected document - Word', re.compile('^Allow changes to parts of a protected document\nAllow changes to parts of a protected document\nProtec.{108} document\nFind parts you can edit\nThis modification is not allowed because the selection is locked\\.\n$', flags=48), re.compile('^Allow changes to parts of a protected document \\- Word\nAllow changes to parts of a protected document.{14374} document, workbook, or presentation\nA subscription to make the most of your time\nTry one month free$', flags=48)),
            9: CoreDocument('msmarco_doc_07_1025413044', 'How to Slow Cook Beef Top Round Roasts | Livestrong.com', re.compile("^How to Slow Cook Beef Top Round Roasts\nHow to Slow Cook Beef Top Round Roasts\nTip\nCrockpot Top Round.{4}st Recipe\nThings You'll Need\nFrequently Asked Questions on Roasts\nMeat Eating Guidelines\nreferences\n$", flags=48), re.compile('^How to Slow Cook Beef Top Round Roasts \\| Livestrong\\.com\nNutrition\nCooking and Baking\nCooking Techniq.{4782}uide to the Leanest Selections"\nU\\.S\\. Department of Agriculture: "10 Tips: Vary Your Protein Routine"$', flags=48)),
            6032: CoreDocument('msmarco_doc_13_766206358', 'which group of players in the game of economics both produces and uses goods and services - Brainly.com', re.compile('^Which group of players in the game of economics both produces and uses goods and services\nWhich grou.{210}estion\\?\nStill have questions\\?\nNew questions in Social Studies\nGet more out of Brainly\nJoin for free\n$', flags=48), re.compile('^which group of players in the game of economics both produces and uses goods and services \\- Brainly\\..{4016}our own questions\ncheck\nLearn with a massive community\nJoin for free\nAlready have an account\\? Log in$', flags=48)),
            10000: CoreDocument('msmarco_doc_05_1281134085', 'Difference Between HRM and Personnel management | Difference Between', 'Difference Between HRM and Personnel management\nDifference Between HRM and Personnel management\nHRM vs Personnel management\nPrabhat S\nSearch DifferenceBetween.net :\nLeave a Response\n', re.compile('^Difference Between HRM and Personnel management \\| Difference Between\nDifference Between HRM and Pers.{3982}ser assumes all risk of use, damage, or injury\\. You agree that we have no liability for any damages\\.$', flags=48)),
            99999: CoreDocument('msmarco_doc_43_202901511', 'How to Fix (and Adjust) Automatic Updating in Google Chrome', re.compile('^How to Fix \\(and Adjust\\) Automatic Updating in Google Chrome\nHow to Fix \\(and Adjust\\) Automatic Updati.{444}low Installations\nGoogle Update \\-> Applications \\-> Google Chrome Binaries \\-> Update Policy Override\n$', flags=48), re.compile('^How to Fix \\(and Adjust\\) Automatic Updating in Google Chrome\nHow to Fix \\(and Adjust\\) Automatic Updati.{10221}nd How Do You Use It\\?\nThe above article may contain affiliate links, which help support How\\-To Geek\\.$', flags=48)),
        })
        self._test_docs('core/document/1M', count=1000000, items={
            0: CoreDocument('msmarco_doc_29_356827432', 'Allow changes to parts of a protected document - Word', re.compile('^Allow changes to parts of a protected document\nAllow changes to parts of a protected document\nProtec.{108} document\nFind parts you can edit\nThis modification is not allowed because the selection is locked\\.\n$', flags=48), re.compile('^Allow changes to parts of a protected document \\- Word\nAllow changes to parts of a protected document.{14374} document, workbook, or presentation\nA subscription to make the most of your time\nTry one month free$', flags=48)),
            9: CoreDocument('msmarco_doc_07_1025413044', 'How to Slow Cook Beef Top Round Roasts | Livestrong.com', re.compile("^How to Slow Cook Beef Top Round Roasts\nHow to Slow Cook Beef Top Round Roasts\nTip\nCrockpot Top Round.{4}st Recipe\nThings You'll Need\nFrequently Asked Questions on Roasts\nMeat Eating Guidelines\nreferences\n$", flags=48), re.compile('^How to Slow Cook Beef Top Round Roasts \\| Livestrong\\.com\nNutrition\nCooking and Baking\nCooking Techniq.{4782}uide to the Leanest Selections"\nU\\.S\\. Department of Agriculture: "10 Tips: Vary Your Protein Routine"$', flags=48)),
            6032: CoreDocument('msmarco_doc_13_766206358', 'which group of players in the game of economics both produces and uses goods and services - Brainly.com', re.compile('^Which group of players in the game of economics both produces and uses goods and services\nWhich grou.{210}estion\\?\nStill have questions\\?\nNew questions in Social Studies\nGet more out of Brainly\nJoin for free\n$', flags=48), re.compile('^which group of players in the game of economics both produces and uses goods and services \\- Brainly\\..{4016}our own questions\ncheck\nLearn with a massive community\nJoin for free\nAlready have an account\\? Log in$', flags=48)),
            10000: CoreDocument('msmarco_doc_05_1281134085', 'Difference Between HRM and Personnel management | Difference Between', 'Difference Between HRM and Personnel management\nDifference Between HRM and Personnel management\nHRM vs Personnel management\nPrabhat S\nSearch DifferenceBetween.net :\nLeave a Response\n', re.compile('^Difference Between HRM and Personnel management \\| Difference Between\nDifference Between HRM and Pers.{3982}ser assumes all risk of use, damage, or injury\\. You agree that we have no liability for any damages\\.$', flags=48)),
            100000: CoreDocument('msmarco_doc_04_272426581', 'Inside Apple HQ - Apple Gazette', re.compile('^Inside Apple HQ\nInside Apple HQ\nOutside the main entrance at Apple Headquarters\nCHANGE LOG:\nAtriums .{158}nter\nTown Hall\nLabs\nApple HQ Industrial Design Lab\nRecreation\nBandley Fitness Center\nParking Garage\n$', flags=48), re.compile('^Inside Apple HQ \\- Apple Gazette\nInside Apple HQ\nApril 5, 2012 by Robin Parrish 85 Comments\nOutside t.{21109} at Apple HQ\n\\[ Image source\\]\n\\[ Header image source\\]\nSpecial thanks to our web host uswebhosting\\.com\\.$', flags=48)),
            999999: CoreDocument('msmarco_doc_46_1140479816', 'Code of Chivalry', re.compile('^Knights Code of Chivalry\nKnights Code of Chivalry\nCode of Chivalry Definition\nKnights Code of Chival.{209}lry and Christianity\nSummary of Chivalry\nThe end of chivalry\nLearn More about Chivalry at Wikipedia\n$', flags=48), re.compile('^Code of Chivalry\n\n\n\nMedieval Chronicles /\nMedieval Knights /\nKnights Code of Chivalry\n/\nKnights Code.{8532}ieval Knights Helmets\nHome\nAbout Us\nContact Us\nSitemap\nCopyright \\- 2014 \\- 2021 \\- Medieval Chronicles$', flags=48)),
        }, test_docstore=False, test_iter_split=False)
        self._test_docs('core/document', count=10000000, items={
            0: CoreDocument('msmarco_doc_29_356827432', 'Allow changes to parts of a protected document - Word', re.compile('^Allow changes to parts of a protected document\nAllow changes to parts of a protected document\nProtec.{108} document\nFind parts you can edit\nThis modification is not allowed because the selection is locked\\.\n$', flags=48), re.compile('^Allow changes to parts of a protected document \\- Word\nAllow changes to parts of a protected document.{14374} document, workbook, or presentation\nA subscription to make the most of your time\nTry one month free$', flags=48)),
            9: CoreDocument('msmarco_doc_07_1025413044', 'How to Slow Cook Beef Top Round Roasts | Livestrong.com', re.compile("^How to Slow Cook Beef Top Round Roasts\nHow to Slow Cook Beef Top Round Roasts\nTip\nCrockpot Top Round.{4}st Recipe\nThings You'll Need\nFrequently Asked Questions on Roasts\nMeat Eating Guidelines\nreferences\n$", flags=48), re.compile('^How to Slow Cook Beef Top Round Roasts \\| Livestrong\\.com\nNutrition\nCooking and Baking\nCooking Techniq.{4782}uide to the Leanest Selections"\nU\\.S\\. Department of Agriculture: "10 Tips: Vary Your Protein Routine"$', flags=48)),
            6032: CoreDocument('msmarco_doc_13_766206358', 'which group of players in the game of economics both produces and uses goods and services - Brainly.com', re.compile('^Which group of players in the game of economics both produces and uses goods and services\nWhich grou.{210}estion\\?\nStill have questions\\?\nNew questions in Social Studies\nGet more out of Brainly\nJoin for free\n$', flags=48), re.compile('^which group of players in the game of economics both produces and uses goods and services \\- Brainly\\..{4016}our own questions\ncheck\nLearn with a massive community\nJoin for free\nAlready have an account\\? Log in$', flags=48)),
            10000: CoreDocument('msmarco_doc_05_1281134085', 'Difference Between HRM and Personnel management | Difference Between', 'Difference Between HRM and Personnel management\nDifference Between HRM and Personnel management\nHRM vs Personnel management\nPrabhat S\nSearch DifferenceBetween.net :\nLeave a Response\n', re.compile('^Difference Between HRM and Personnel management \\| Difference Between\nDifference Between HRM and Pers.{3982}ser assumes all risk of use, damage, or injury\\. You agree that we have no liability for any damages\\.$', flags=48)),
            100000: CoreDocument('msmarco_doc_04_272426581', 'Inside Apple HQ - Apple Gazette', re.compile('^Inside Apple HQ\nInside Apple HQ\nOutside the main entrance at Apple Headquarters\nCHANGE LOG:\nAtriums .{158}nter\nTown Hall\nLabs\nApple HQ Industrial Design Lab\nRecreation\nBandley Fitness Center\nParking Garage\n$', flags=48), re.compile('^Inside Apple HQ \\- Apple Gazette\nInside Apple HQ\nApril 5, 2012 by Robin Parrish 85 Comments\nOutside t.{21109} at Apple HQ\n\\[ Image source\\]\n\\[ Header image source\\]\nSpecial thanks to our web host uswebhosting\\.com\\.$', flags=48)),
            1000000: CoreDocument('msmarco_doc_26_1948926872', 'Safe Driving Practices Flashcards | Quizlet', 'Safe Driving Practices\nSafe Driving Practices\nSTUDY\nPLAY\nTerms in this set (50)\n', re.compile('^Safe Driving Practices Flashcards \\| Quizlet\nSafe Driving Practices\nSTUDY\nFlashcards\nLearn\nWrite\nSpel.{8965}a distraction that increases your risk of having an accident\\.\nUpgrade to remove ads\nOnly \\$2\\.99/month$', flags=48)),
            9999999: CoreDocument('msmarco_doc_58_240473545', 'How to Paste on Facebook Messenger (with Pictures) - wikiHow', re.compile('^How to Paste on Facebook Messenger\nHow to Paste on Facebook Messenger\nMethod 1\nof 2:\nPasting in Face.{116}top Download Article\nPRO\nCommunity Q\\&A\nTips\nWarnings\nAbout This Article\nIs this article up to date\\?\n$', flags=48), re.compile('^How to Paste on Facebook Messenger \\(with Pictures\\) \\- wikiHow\nCategories\nComputers and Electronics\nIn.{13649}anks to all authors for creating a page that has been read 30,362 times\\.\nIs this article up to date\\?$', flags=48)),
        }, test_docstore=False, test_iter_split=False)

    def test_document_queries(self):
        self._test_queries('core/document', count=55, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2005952', 'what causes eye blackouts'),
            54: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/document/core', count=55, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2005952', 'what causes eye blackouts'),
            54: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/document/10k', count=55, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2005952', 'what causes eye blackouts'),
            54: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/document/100k', count=55, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2005952', 'what causes eye blackouts'),
            54: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })
        self._test_queries('core/document/1M', count=55, items={
            0: GenericQuery('2001010', 'cost comparison of funerals in australia'),
            9: GenericQuery('2005952', 'what causes eye blackouts'),
            54: GenericQuery('3100918', 'Can older adults gain strength by training once per week?'),
        })

    def test_document_qrels(self):
        self._test_qrels('core/document', count=6050, items={
            0: TrecQrel('2001010', 'msmarco_doc_00_1413652624', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_doc_41_285574609', 1, '0'),
            6049: TrecQrel('3100918', 'msmarco_doc_56_1665599430', 0, '0'),
        })
        self._test_qrels('core/document/core', count=6050, items={
            0: TrecQrel('2001010', 'msmarco_doc_00_1413652624', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_doc_41_285574609', 1, '0'),
            6049: TrecQrel('3100918', 'msmarco_doc_56_1665599430', 0, '0'),
        })
        self._test_qrels('core/document/10k', count=6050, items={
            0: TrecQrel('2001010', 'msmarco_doc_00_1413652624', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_doc_41_285574609', 1, '0'),
            6049: TrecQrel('3100918', 'msmarco_doc_56_1665599430', 0, '0'),
        })
        self._test_qrels('core/document/100k', count=6050, items={
            0: TrecQrel('2001010', 'msmarco_doc_00_1413652624', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_doc_41_285574609', 1, '0'),
            6049: TrecQrel('3100918', 'msmarco_doc_56_1665599430', 0, '0'),
        })
        self._test_qrels('core/document/1M', count=6050, items={
            0: TrecQrel('2001010', 'msmarco_doc_00_1413652624', 1, '0'),
            9: TrecQrel('2001010', 'msmarco_doc_41_285574609', 1, '0'),
            6049: TrecQrel('3100918', 'msmarco_doc_56_1665599430', 0, '0'),
        })


if __name__ == '__main__':
    unittest.main()
