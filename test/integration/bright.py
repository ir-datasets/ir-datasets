import re
import unittest
import re
from ir_datasets.datasets.bright import BrightQuery
from ir_datasets.formats import TrecQrel, GenericDoc
from .base import DatasetIntegrationTest


class TestBright(DatasetIntegrationTest):
    def test_docs(self):
        self._test_docs('bright/aops', count=188002, items={
            0: GenericDoc('TheoremQA_xinyi/concavity.json', re.compile('^Consider a source X with a distortion measure \\$d\\(x, \\\\hat\\{x\\}\\)\\$ that satisfies the following property:.{119}\\\\max_\\{b:\\\\sum_\\{i=1\\}\\^m p_i d_i \\\\leq D\\} H\\(p\\)\\$ is concave\\. True or False\\?\nTherefore, the answer is True\\.$', flags=48)),
            9: GenericDoc('TheoremQA_elainewan/math_calculus_2.json', re.compile('^What is \\\\lim_\\{x \\\to 9\\} \\(\\(x \\- 9\\)/\\(\\\\sqrt\\{x\\} \\- 3\\)\\)\\?\nWe can directly substitute the value of x as 9 in th.{483}tituting x as 9, we get:\n\nlim_\\{x \\-> 9\\} \\(\\\\sqrt\\{x\\} \\+ 3\\) = sqrt\\(9\\) \\+ 3 = 6\n\nTherefore, the answer is 6\\.$', flags=48)),
            188001: GenericDoc('aops_2017_AIME_II_Problems/Problem_3', re.compile('^A triangle has vertices \\$A\\(0,0\\)\\$, \\$B\\(12,0\\)\\$, and \\$C\\(8,10\\)\\$\\. The probability that a randomly chosen p.{1820} of \\$ABC\\$, which is \\$\\\\frac\\{\\\\frac\\{109\\}\\{5\\}\\}\\{60\\}=\\\\frac\\{109\\}\\{300\\}\\$\\. The answer is \\$109\\+300=\\\\boxed\\{409\\}\\$\\.$', flags=48)),
        })

        self._test_docs('bright/biology', count=57359, items={
            0: GenericDoc('neanderthals_vitamin_C_diet/Neanderthal_0_43.txt', re.compile('^ pelvises; and proportionally shorter forearms and forelegs\\.\nBased on 45 Neanderthal long bones from.{187}20 males and 10 females Upper Palaeolithic humans is, respectively, 176\\.2\xa0cm \\(5\xa0ft 9\\.4\xa0in\\) and 162\\.9$', flags=48)),
            9: GenericDoc('neanderthals_vitamin_C_diet/Neanderthal_0_83.txt', re.compile('^ trends seen in obsidian transfer distance and tribe size in modern hunter\\-gatherers\\. However, accor.{420}nufacturing and so on\\. La Ferrassie is also located in one of the richest animal\\-migration routes of$', flags=48)),
            57358: GenericDoc('hot_water_bacteria/Thehandiworkofgoodhe_20_0.txt', "Don't scrub. Scrubbing can damage skin, especially if you do it a lot. The resulting cracks and small cuts give pathogens a place to grow."),
        })

        self._test_docs('bright/earth-science', count=121249, items={
            0: GenericDoc('arid_area/Earth_rainfall_climatology4_10.txt', re.compile('^, tundra, polar ice cap, and desert\\.\nRain forests are characterized by high rainfall, with definitio.{256}etween 750 millimetres \\(30\xa0in\\) and 1,270 millimetres \\(50\xa0in\\) a year\\.  They are widespread on Africa,$', flags=48)),
            9: GenericDoc('arid_area/Earth_rainfall_climatology4_27.txt', re.compile('^ are also possible across Atlantic Canada\\.  Amounts decrease as one works farther inland from the Pa.{348}ese regions averages between 300 and 600 millimeters \\(11\\.8 and 23\\.6\xa0in\\) per year, with lower amounts$', flags=48)),
            121248: GenericDoc('newton_estimation/ReviewofGravity_html4_2.txt', re.compile("^ is about 10 times the radius of the earth\\. This means the object is times farther from the center o.{327}ter reading this section, it is recommended\nto check the following  movie of Kepler's laws\\. \nhttp://$", flags=48)),
        })

        self._test_docs('bright/economics', count=50220, items={
            0: GenericDoc('uspopulationgrowth/thelongtermdeclineinfertilityandwhatitmeansforstatebudgets_76.txt', re.compile('^Changing societal norms and other factors have reshaped recent generations of\nAmerican families\\. Fol.{402}andemic—showing 43 states\nrecorded their lowest general fertility rate in at least three decades\\.  7$', flags=48)),
            9: GenericDoc('uspopulationgrowth/ImmigrantsComingAmericaOlderAges_7.txt', re.compile('^population has aged significantly in the last two decades\\.\n\\*\\*The Increasing Age of all Immigrants\\.\\*\\*.{3760}\\. Moreover, immigrants\nare now arriving at older ages compared to newcomers in previous years\\. These$', flags=48)),
            50219: GenericDoc('chi_squared/Chisquareddistribution_72.txt', re.compile('^χ  2  \\{\\\\displaystyle \\\\chi \\^\\{2\\}\\}  !\\[\\{\\\\\\\\displaystyle \\\\\\\\chi\n\\^\\{2\\}\\}\\]\\(https://wikimedia\\.org/api/rest_v1/me.{1567} the table\\.  \\[ _\\[ citation needed\n\\]\\(/wiki/Wikipedia:Citation_needed "Wikipedia:Citation needed"\\) _ \\]$', flags=48)),
        })

        self._test_docs('bright/leetcode', count=413932, items={
            0: GenericDoc('leetcode/csn_python_train_152036.txt', re.compile("^def get\\(tgt,\n        fun,\n        tgt_type='glob',\n        exclude_minion=False\\):\n    '''\n    Get da.{3193}exclude_minion:\n        if __opts__\\['id'\\] in ret:\n            del ret\\[__opts__\\['id'\\]\\]\n    return ret$", flags=48)),
            9: GenericDoc('leetcode/csn_python_train_323590.txt', re.compile('^def xstep\\(self\\):\n        r"""Minimise Augmented Lagrangian with respect to\n        :math:`\\\\mathbf\\{x\\}.{135}\\) \\+\n            self\\.D\\.T\\.dot\\(self\\.block_sep1\\(YU\\)\\), self\\.lu, self\\.piv\\),\n            dtype=self\\.dtype\\)$', flags=48)),
            413931: GenericDoc('leetcode/csn_python_train_35369.txt', re.compile('^def get_user_groups\\(self, user_name\\):\n        """Get a list of groups associated to a user\\.\n\n       .{458}     return \\[group\\.text for group in tree\\.find\\(\'data/groups\'\\)\\]\n\n        raise HTTPResponseError\\(res\\)$', flags=48)),
        })

        self._test_docs('bright/pony', count=7894, items={
            0: GenericDoc('Pony/src-builtin-runtime_options-_1.txt', '12\n13\n14\n15\n16\n17\n18\n19\n20\n21\n22\n23\n24\n25\n26\n27\n28\n29\n30\n31'),
            9: GenericDoc('Pony/builtin-Array-_2.txt', re.compile('^\nModifying the Array can be done via update, insert and delete methods\nwhich alter the Array at an a.{820}quested number of elements and allocate that much space, with a\nlower bound of space for 8 elements\\.$', flags=48)),
            7893: GenericDoc('Pony/builtin-FloatingPoint-_53.txt', '\ny: A\n\nReturns¶\n\nA\n\n\nmax¶\n[Source]\nfun box max(\ny: A)\n: A\n\nParameters¶\n\ny: A\n\nReturns¶\n'),
        })

        self._test_docs('bright/psychology', count=52835, items={
            0: GenericDoc('horn_effect/attractivepeoplegetunfairadvantagesatworkaicanhelp_65.txt', re.compile('^\\[ Latest \\]\\(/the\\-latest\\) \\[ Magazine \\]\\(/magazine\\) \\[ Ascend \\]\\(/ascend\\) \\[ Topics\n\\]\\(/topics\\) \\[ Reading Li.{92} \\[ Data \\& Visuals \\]\\(/data\\-visuals\\) \\[\nCase Selections \\]\\(/case\\-selections\\) \\[ HBR Learning \\]\\(/learning\\)$', flags=48)),
            9: GenericDoc('horn_effect/Physicalattractivenessstereotype_21.txt', '5 languages'),
            52834: GenericDoc('quasi_experiment/quasiexperimentalresearchtypesexamplesapplication_0.txt', "[ ](/) [ Log in ](/auth?signin='true')"),
        })

        self._test_docs('bright/robotics', count=61961, items={
            0: GenericDoc('detachable_joint/en_187.txt', 'Translated from  German'),
            9: GenericDoc('detachable_joint/en_139.txt', 'UmbÃ¶rdelung  \nFlanging  '),
            61960: GenericDoc('can_message/Software680htmlL1gad_96.txt', '[ ![](typo3temp/pics/76676042ee.jpg) ](PCAN-FMS-Simulator-2.246.0.html?&L=1)'),
        })

        self._test_docs('bright/stackoverflow', count=107081, items={
            0: GenericDoc('R_base_tools/R_base_all_234_0.txt', re.compile('^\\[ RDocumentation \\]\\(/\\)\n\nMoon  \\[ \\]\\(https://github\\.com/datacamp/rdocumentation\\-2\\.0\\) \\[ Learn R\n\\]\\(https:/.{2229}ansform\\&utm_content=run_example_in_workspace\\)\n\nPowered by  \\[ DataCamp \\]\\(https://www\\.datacamp\\.com/\\)\n\n$', flags=48)),
            9: GenericDoc('R_base_tools/R_base_all_188_1.txt', re.compile('^ction\\.\n\nBecause of the way it is implemented, on a Unix\\-alike ` stderr = TRUE `\nimplies ` stdout = T.{813}nk/shell\\.exec\\?package=base\\&version=3\\.6\\.2\\) ` \\.\n\nPowered by  \\[ DataCamp \\]\\(https://www\\.datacamp\\.com/\\)\n\n$', flags=48)),
            107080: GenericDoc('Mmdn_RegExp/exec_48_0.txt', 'Get real-time assistance and support'),
        })

        self._test_docs('bright/sustainable-living', count=60792, items={
            0: GenericDoc('mosquito_control/447_60.txt', 'All fields  Title  Abstract  Keywords  Authors  Affiliations  Doi  Full Text\nReferences'),
            9: GenericDoc('mosquito_control/447_210.txt', re.compile('^\\*\\*Figure 8\\.\\*\\* Installing of single\\-charged dipolar electric field screens to\nlateral windows \\( \\*\\*A\\*\\*.{16}e greenhouse and an electric circuit to\nlink the screens \\( \\*\\*C\\*\\* \\) to a voltage generator \\( \\*\\*D\\*\\* \\)\\.$', flags=48)),
            60791: GenericDoc('bokashi/whatisbokashibran_55.txt', '##  Why is Bokashi Bran effective at fermenting food waste?'),
        })

        self._test_docs('bright/theoremqa-questions', count=188002, items={
            0: GenericDoc('TheoremQA_xinyi/concavity.json', re.compile('^Consider a source X with a distortion measure \\$d\\(x, \\\\hat\\{x\\}\\)\\$ that satisfies the following property:.{119}\\\\max_\\{b:\\\\sum_\\{i=1\\}\\^m p_i d_i \\\\leq D\\} H\\(p\\)\\$ is concave\\. True or False\\?\nTherefore, the answer is True\\.$', flags=48)),
            9: GenericDoc('TheoremQA_elainewan/math_calculus_2.json', re.compile('^What is \\\\lim_\\{x \\\to 9\\} \\(\\(x \\- 9\\)/\\(\\\\sqrt\\{x\\} \\- 3\\)\\)\\?\nWe can directly substitute the value of x as 9 in th.{483}tituting x as 9, we get:\n\nlim_\\{x \\-> 9\\} \\(\\\\sqrt\\{x\\} \\+ 3\\) = sqrt\\(9\\) \\+ 3 = 6\n\nTherefore, the answer is 6\\.$', flags=48)),
            188001: GenericDoc('aops_2017_AIME_II_Problems/Problem_3', re.compile('^A triangle has vertices \\$A\\(0,0\\)\\$, \\$B\\(12,0\\)\\$, and \\$C\\(8,10\\)\\$\\. The probability that a randomly chosen p.{1820} of \\$ABC\\$, which is \\$\\\\frac\\{\\\\frac\\{109\\}\\{5\\}\\}\\{60\\}=\\\\frac\\{109\\}\\{300\\}\\$\\. The answer is \\$109\\+300=\\\\boxed\\{409\\}\\$\\.$', flags=48)),
        })

        self._test_docs('bright/theoremqa-theorems', count=23839, items={
            0: GenericDoc('0', "\\begin{definition}[Definition:Addition]\n'''Addition''' is the basic operation $+$ everyone is familiar with.\nFor example:\n:$2 + 3 = 5$\n:$47 \\cdotp 3 + 191\\cdotp 4 = 238 \\cdotp 7$\n\\end{definition}"),
            9: GenericDoc('9', re.compile('^\\\\begin\\{definition\\}\\[Definition:Addition/Real Numbers\\]\nThe addition operation in the domain of real nu.{412}\\{x_n\\} \\} \\{\\} \\+ \\\\eqclass \\{\\\\sequence \\{y_n\\} \\} \\{\\} = \\\\eqclass \\{\\\\sequence \\{x_n \\+ y_n\\} \\} \\{\\}\\$\n\\\\end\\{definition\\}$', flags=48)),
            23838: GenericDoc('23838', re.compile("^\\\\section\\{Non\\-Archimedean Division Ring Iff Non\\-Archimedean Completion\\}\nTags: Complete Metric Spaces,.{1216}rm \\{\\\\, \\\\cdot \\\\,\\}' \\}\\$ non\\-archimedean \\{\\{iff\\}\\} \\$\\\\struct \\{R', \\\\norm \\{\\\\, \\\\cdot \\\\,\\}' \\}\\$ is\\.\n\\\\end\\{proof\\}\n\n$", flags=48)),
        })

        self._test_docs('bright/biology-long', count=524, items={
            0: GenericDoc('protein_in_food/Protein_folding.txt', re.compile('^Protein folding is the physical process by which a protein, after synthesis by a ribosome as a linea.{37857}clic amplification\nProtein structure prediction software\nProteopathy\nTime\\-resolved mass spectrometry$', flags=48)),
            9: GenericDoc('adults_lose_hearing/Sound.txt', re.compile('^\nIn physics, sound is a vibration that propagates as an acoustic wave through a transmission medium .{21951}sical tone\nResonance\nReverberation\nSonic weaponry\nSound synthesis\nSoundproofing\nStructural acoustics$', flags=48)),
            523: GenericDoc('UV_causes_skin_cancer/uva-vs-uvb.txt', re.compile('^\n\nHealth Conditions Featured Breast Cancer IBD Migraine Multiple Sclerosis \\(MS\\) Rheumatoid Arthritis.{128321}de medical advice, diagnosis, or treatment\\. See additional information\\. See additional information \\.$', flags=48)),
        })

        self._test_docs('bright/earth-science-long', count=601, items={
            0: GenericDoc('plant_produce_oxygen/Atmosphere_of_Earth.txt', re.compile('^\n\n\n\nAtmosphere of Earth \\- Wikipedia\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nJump to content\n\n\n\n\n\n\n\nMain m.{57540}evelopers\nStatistics\nCookie statement\nMobile view\n\n\n\n\n\n\n\n\n\n\n\n\n\nToggle limited content width\n\n\n\n\n\n\n\n\n$', flags=48)),
            9: GenericDoc('sendimentation_without_water/Tuff.txt', re.compile('^\n\n\n\nTuff \\- Wikipedia\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nJump to content\n\n\n\n\n\n\n\nMain menu\n\n\n\n\n\nMain .{37844}evelopers\nStatistics\nCookie statement\nMobile view\n\n\n\n\n\n\n\n\n\n\n\n\n\nToggle limited content width\n\n\n\n\n\n\n\n\n$', flags=48)),
            600: GenericDoc('moon_collided_with_earth/Giant-impact_hypothesis.txt', re.compile('^The giant\\-impact hypothesis, sometimes called the Big Splash, or the Theia Impact, is an astrogeolog.{24170}tory of Earth\nLunar geologic timescale\nOrigin of the Moon\nRoche limit\nPhaeton \\(hypothetical planet\\)\n$', flags=48)),
        })

        self._test_docs('bright/economics-long', count=516, items={
            0: GenericDoc('russia_rich/waragainstukrainewhatifrussiawins.txt', re.compile('^Skip to main content\n\n\\[ !\\[GLOBSEC \\- A Global Think Tank: Ideas Shaping the World\nlogo\\]\\(/themes/custo.{18751}rship\\)\n\nCopyright © 2008\\-2024 GLOBSEC \\| All rights reserved\n\n\\* \\[ Privacy policy \\]\\(/privacy\\-policy\\)\n\n$', flags=48)),
            9: GenericDoc('economic_growth/Balancedgrowthequilibrium.txt', re.compile('^Jump to content\n\nMain menu\n\nMain menu\n\nmove to sidebar  hide\n\nNavigation\n\n\\* \\[ Main page  \\]\\(/wiki/Mai.{11796}content width\n\n\\*\\[\nv\n\\]: View this template\n\\*\\[\nt\n\\]: Discuss this template\n\\*\\[\ne\n\\]: Edit this template\n\n$', flags=48)),
            515: GenericDoc('public_debt_default/debitopubblicohtml.txt', re.compile('^\\#  Quick access menu to the contents of the Treasury Department website\n\nGo to the main navigation m.{13365}Inflation\n\\*\\[\nBTPs Italia\n\\]: Treasury Bonds Linked to Italian Inflation\n\\*\\[\nMTN\n\\]: Medium Term Notes\n\n$', flags=48)),
        })

        self._test_docs('bright/pony-long', count=577, items={
            0: GenericDoc('Pony/pony_bench-BenchmarkList-.txt', re.compile('^\n\n\n\n\n\n\nBenchmarkList¶\n\\[Source\\]\ninterface tag BenchmarkList\n\nPublic Functions¶\nbenchmarks¶\n\\[Source\\]\nf.{23}h: PonyBench tag\\)\n: None val\n\nParameters¶\n\nbench: PonyBench tag\n\nReturns¶\n\nNone val\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n$', flags=48)),
            9: GenericDoc('Pony/builtin-Equal-.txt', re.compile('^\n\n\n\n\n\n\nEqual¶\n\\[Source\\]\nprimitive val Equal is\nEquatable\\[\\(Less val \\| Equal val \\| Greater val\\)\\] ref\n\nI.{474}ol val\n\nParameters¶\n\nthat: \\(Less val \\| Equal val \\| Greater val\\)\n\nReturns¶\n\nBool val\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n$', flags=48)),
            576: GenericDoc('Pony/builtin-Stringable-.txt', re.compile('^\n\n\n\n\n\n\nStringable¶\n\\[Source\\]\nThings that can be turned into a String\\.\ninterface box Stringable\n\nPubli.{49}epresentation of this object\\.\nfun box string\\(\\)\n: String iso\\^\n\nReturns¶\n\nString iso\\^\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n$', flags=48)),
        })

        self._test_docs('bright/psychology-long', count=512, items={
            0: GenericDoc('hindsight_bias/Barnumeffect.txt', re.compile('^Jump to content\n\nMain menu\n\nMain menu\n\nmove to sidebar  hide\n\nNavigation\n\n\\* \\[ Main page  \\]\\(/wiki/Mai.{36319}content width\n\n\\*\\[\nv\n\\]: View this template\n\\*\\[\nt\n\\]: Discuss this template\n\\*\\[\ne\n\\]: Edit this template\n\n$', flags=48)),
            9: GenericDoc('russell/whyisitsohardtochangepeoplesminds.txt', re.compile('^SKIP TO:  Header  \\| \\[ Log in \\]\\(https://greatergood\\.berkeley\\.edu/_ajax/login\\-\npopup\\) \\| \\[ Register \\]\\(h.{25545}e” to you and to millions around the globe\\.\n\n\\[ Give Now \\]\\(https://greatergood\\.berkeley\\.edu/donate\\)\n\n$', flags=48)),
            511: GenericDoc('emotion_heart/anahatachakraunlockingtheheartchakra.txt', re.compile('^\nHome Gallery About Us Classes \\& Events Blog Contact Us Members Area\nANAHATA CHAKRA \\- unlocking the .{11536} 2024\nAll Rights Reserved \\| YogaOne Australia\n\nPrivacy Policy\n\nPowered by BGR Development\n\nShare by:$', flags=48)),
        })

        self._test_docs('bright/robotics-long', count=508, items={
            0: GenericDoc('number_commands/24301.txt', re.compile('^  \nROS Resources: \\[ ROS Homepage \\]\\(http://ros\\.org/\\) \\| \\[ Media and Trademarks\n\\]\\(https://www\\.ros\\.org/b.{11331}rivacy\\)\n\nPowered by \\[ Discourse \\]\\(https://www\\.discourse\\.org\\) , best viewed with\nJavaScript enabled\n\n$', flags=48)),
            9: GenericDoc('takeoff_rotation/16374.txt', re.compile('^\\[ The Construct ROS Community \\]\\(/\\)\n\n\\#  \\[ How to hover a drone at certian height\\? \\]\\(/t/how\\-to\\-hover\\-a.{14162}olicy/\\)\n\nPowered by \\[ Discourse \\]\\(https://www\\.discourse\\.org\\) , best viewed with\nJavaScript enabled\n\n$', flags=48)),
            507: GenericDoc('ros_yaml/rclpyparamstutorialg.txt', re.compile('^Skip to content\n\n\\[ !\\[The Robotics Back\\-End\\]\\(https://roboticsbackend\\.com/wp\\-\ncontent/uploads/2020/02/.{23046}/ros2\\-for\\-beginners\\)\n\n\\*\\*\\[ Check out the course here \\]\\(https://rbcknd\\.com/ros2\\-for\\-beginners\\) \\*\\*\n\n×\n\n$', flags=48)),
        })

        self._test_docs('bright/stackoverflow-long', count=1858, items={
            0: GenericDoc('DBMS_LOB_LIBCACHE/oracle_LIBCACHE.txt', re.compile('^\\[ Previous \\]\\(DBMS_LDAP_UTL\\.html\\) \\[ Next  \\]\\(DBMS_LOB\\.html\\) JavaScript must be\nenabled to correctly di.{5874}arsing\\. This parameter is reserved for parallel compile jobs which are currently not implemented\\. \n\n$', flags=48)),
            9: GenericDoc('NET_Methods_Properties/NET_Methods.txt', re.compile('^Skip to main content\n\nThis browser is no longer supported\\.\n\nUpgrade to Microsoft Edge to take advant.{243415}ademarks \\]\\(https://www\\.microsoft\\.com/legal/intellectualproperty/Trademarks/\\)\n  \\* © Microsoft 2024 \n\n$', flags=48)),
            1857: GenericDoc('fastapi_security_advanced_user_guide/configuration.txt', re.compile('^\\[ !\\[Swagger\nLogo\\]\\(https://static0\\.smartbear\\.co/swagger/media/assets/images/swagger_logo\\.svg\\)\n\\]\\(/\\)\n\n\\*.{22702}https://www\\.linkedin\\.com/company/smartbear/\\) \\[ __\n\\]\\(http://www\\.youtube\\.com/user/SmartBearSoftware\\)\n\n$', flags=48)),
        })

        self._test_docs('bright/sustainable-living-long', count=554, items={
            0: GenericDoc('asparagas/Permaculture.txt', re.compile('^Jump to content\n\nMain menu\n\nMain menu\n\nmove to sidebar  hide\n\nNavigation\n\n\\* \\[ Main page  \\]\\(/wiki/Mai.{170361}content width\n\n\\*\\[\nv\n\\]: View this template\n\\*\\[\nt\n\\]: Discuss this template\n\\*\\[\ne\n\\]: Edit this template\n\n$', flags=48)),
            9: GenericDoc('foam/whatkindofpolyethylenematerialcanberecycledhtml.txt', re.compile('^Select Language\nHOME\nABOUT\nCOMPACT\nMELT\nDEWATER\nCASE\nVIDEO\nNEWS\nCONTACT US\nWhat kind of polyethylene.{3321}ill send detailed info and quotation\n\n\nName\n\\*\nEmail\n\\*\nTelephone\nCompany Name\n\\*\nMessage\n\\*\nContact Us\n$', flags=48)),
            553: GenericDoc('fresh_air_car/useherbsasanallnaturalairfreshener5944860.txt', re.compile('^Skip to Main Content\n\n\\[ !\\[Lifehacker Logo\\]\\(/images/lifehacker\\-logo\\.svg\\) \\]\\(https://lifehacker\\.com\\)\n\n\\*.{10783}\\\\\\(opens in a new tab\\\\\\)"\\)\n\\* \\[ Speedtest Logo  \\]\\(https://www\\.speedtest\\.net "\\\\\\(opens in a new tab\\\\\\)"\\)\n\n$', flags=48)),
        })

    def test_queries(self):
        self._generate_test_queries('bright/aops')
        self._generate_test_queries('bright/biology')
        self._generate_test_queries('bright/earth-science')
        self._generate_test_queries('bright/economics')
        self._generate_test_queries('bright/leetcode')
        self._generate_test_queries('bright/pony')
        self._generate_test_queries('bright/psychology')
        self._generate_test_queries('bright/robotics')
        self._generate_test_queries('bright/stackoverflow')
        self._generate_test_queries('bright/sustainable-living')
        self._generate_test_queries('bright/theoremqa-questions')
        self._generate_test_queries('bright/theoremqa-theorems')
        self._generate_test_queries('bright/biology-long')
        self._generate_test_queries('bright/earth-science-long')
        self._generate_test_queries('bright/economics-long')
        self._generate_test_queries('bright/pony-long')
        self._generate_test_queries('bright/psychology-long')
        self._generate_test_queries('bright/robotics-long')
        self._generate_test_queries('bright/stackoverflow-long')
        self._generate_test_queries('bright/sustainable-living-long')

    def test_qrels(self):
        self._test_qrels('bright/aops', count=623280, items={
            0: TrecQrel('aops_2019_AMC_12A_Problems/Problem_17', 'math_test_intermediate_algebra_1179', 1, '0'),
            9: TrecQrel('aops_2019_AMC_12A_Problems/Problem_17', 'camel_6', -100, '0'),
            623279: TrecQrel('aops_2017_AIME_II_Problems/Problem_3', 'math_test_geometry_641', -100, '0'),
        })

        self._test_qrels('bright/biology', count=372, items={
            0: TrecQrel('0', 'insects_attracted_to_light/Proximate_and_ultimate_causation_0.txt', 1, '0'),
            9: TrecQrel('1', 'breathe_out_of_one_nostril/Nasal_cycle_2.txt', 1, '0'),
            371: TrecQrel('102', 'yeast_dissolve_in_sugar/Osmosis_0.txt', 1, '0'),
        })

        self._test_qrels('bright/earth-science', count=585, items={
            0: TrecQrel('0', 'number_of_hadley_cell/Hadley_cell2.txt', 1, '0'),
            9: TrecQrel('3', 'solid_inner_core/Earth%27s_inner_core1.txt', 1, '0'),
            584: TrecQrel('115', 'era5_data/Create_and_Delete_Files_and_Directories_from_Windows_Command_Prompt5.txt', 1, '0'),
        })

        self._test_qrels('bright/economics', count=800, items={
            0: TrecQrel('0', 'stole/ajol_6.txt', 1, '0'),
            9: TrecQrel('1', 'stochastic_dominance/Stochasticdominance_4.txt', 1, '0'),
            799: TrecQrel('102', 'country_firm/granular_6.txt', 1, '0'),
        })

        self._test_qrels('bright/leetcode', count=33747, items={
            0: TrecQrel('0', 'leetcode/leetcode_11.txt', 1, '0'),
            9: TrecQrel('0', 'leetcode/leetcode_496.txt', -100, '0'),
            33746: TrecQrel('141', 'leetcode/leetcode_436.txt', -100, '0'),
        })

        self._test_qrels('bright/pony', count=2219, items={
            0: TrecQrel('0', 'Pony/4_control-structures_13.txt', 1, '0'),
            9: TrecQrel('0', 'Pony/4_control-structures_2.txt', 1, '0'),
            2218: TrecQrel('111', 'Pony/1_variables_1.txt', 1, '0'),
        })

        self._test_qrels('bright/psychology', count=692, items={
            0: TrecQrel('0', 'confirmation_bias/seeds_model_11.txt', 1, '0'),
            9: TrecQrel('1', 'meg/Magnetoencephalography_18.txt', 1, '0'),
            691: TrecQrel('100', 'mentalblock/Mentalblock_3.txt', 1, '0'),
        })

        self._test_qrels('bright/robotics', count=520, items={
            0: TrecQrel('0', 'ros2cpp/navigationlaunchpy_20.txt', 1, '0'),
            9: TrecQrel('2', 'gazebo/indexhtml_2.txt', 1, '0'),
            519: TrecQrel('100', 'relative_path/PackagesClientLibrar_6.txt', 1, '0'),
        })

        self._test_qrels('bright/stackoverflow', count=478, items={
            0: TrecQrel('0', 'snowflake_docs/flatten_2_0.txt', 1, '0'),
            9: TrecQrel('1', 'Python_pandas_functions_with_style/DataFrame_93_5.txt', 1, '0'),
            477: TrecQrel('116', 'R_base_all/R_base_all_351_0.txt', 1, '0'),
        })

        self._test_qrels('bright/sustainable-living', count=576, items={
            0: TrecQrel('0', 'incineration/Incineration_62.txt', 1, '0'),
            9: TrecQrel('0', 'incineration/Incineration_92.txt', 1, '0'),
            575: TrecQrel('107', 'boreal_forest/Silvopasture_5.txt', 1, '0'),
        })

        self._test_qrels('bright/theoremqa-questions', count=607140, items={
            0: TrecQrel('TheoremQA_elainewan/math_calculus_5_2.json', 'TheoremQA_elainewan/math_calculus_5.json', 1, '0'),
            9: TrecQrel('TheoremQA_elainewan/math_calculus_5_2.json', 'camel_6003', -100, '0'),
            607139: TrecQrel('TheoremQA_xinyi/huffman_code_3.json', 'TheoremQA_xinyi/huffman_code_3.json', -100, '0'),
        })

        self._test_qrels('bright/theoremqa-theorems', count=151, items={
            0: TrecQrel('TheoremQA_jianyu_xu/pigeonhole_3.json', '18695', 1, '0'),
            9: TrecQrel("TheoremQA_wenhuchen/L'Hôpital_rule1.json", '11592', 1, '0'),
            150: TrecQrel('TheoremQA_wenhuchen/series_convergen2.json', '8392', 1, '0'),
        })

        self._test_qrels('bright/biology-long', count=372, items={
            0: TrecQrel('0', 'insects_attracted_to_light/Proximate_and_ultimate_causation_0.txt', 1, '0'),
            9: TrecQrel('1', 'breathe_out_of_one_nostril/Nasal_cycle_2.txt', 1, '0'),
            371: TrecQrel('102', 'yeast_dissolve_in_sugar/Osmosis_0.txt', 1, '0'),
        })

        self._test_qrels('bright/earth-science-long', count=585, items={
            0: TrecQrel('0', 'number_of_hadley_cell/Hadley_cell2.txt', 1, '0'),
            9: TrecQrel('3', 'solid_inner_core/Earth%27s_inner_core1.txt', 1, '0'),
            584: TrecQrel('115', 'era5_data/Create_and_Delete_Files_and_Directories_from_Windows_Command_Prompt5.txt', 1, '0'),
        })

        self._test_qrels('bright/economics-long', count=800, items={
            0: TrecQrel('0', 'stole/ajol_6.txt', 1, '0'),
            9: TrecQrel('1', 'stochastic_dominance/Stochasticdominance_4.txt', 1, '0'),
            799: TrecQrel('102', 'country_firm/granular_6.txt', 1, '0'),
        })

        self._test_qrels('bright/pony-long', count=2219, items={
            0: TrecQrel('0', 'Pony/4_control-structures_13.txt', 1, '0'),
            9: TrecQrel('0', 'Pony/4_control-structures_2.txt', 1, '0'),
            2218: TrecQrel('111', 'Pony/1_variables_1.txt', 1, '0'),
        })

        self._test_qrels('bright/psychology-long', count=692, items={
            0: TrecQrel('0', 'confirmation_bias/seeds_model_11.txt', 1, '0'),
            9: TrecQrel('1', 'meg/Magnetoencephalography_18.txt', 1, '0'),
            691: TrecQrel('100', 'mentalblock/Mentalblock_3.txt', 1, '0'),
        })

        self._test_qrels('bright/robotics-long', count=520, items={
            0: TrecQrel('0', 'ros2cpp/navigationlaunchpy_20.txt', 1, '0'),
            9: TrecQrel('2', 'gazebo/indexhtml_2.txt', 1, '0'),
            519: TrecQrel('100', 'relative_path/PackagesClientLibrar_6.txt', 1, '0'),
        })

        self._test_qrels('bright/stackoverflow-long', count=478, items={
            0: TrecQrel('0', 'snowflake_docs/flatten_2_0.txt', 1, '0'),
            9: TrecQrel('1', 'Python_pandas_functions_with_style/DataFrame_93_5.txt', 1, '0'),
            477: TrecQrel('116', 'R_base_all/R_base_all_351_0.txt', 1, '0'),
        })

        self._test_qrels('bright/sustainable-living-long', count=576, items={
            0: TrecQrel('0', 'incineration/Incineration_62.txt', 1, '0'),
            9: TrecQrel('0', 'incineration/Incineration_92.txt', 1, '0'),
            575: TrecQrel('107', 'boreal_forest/Silvopasture_5.txt', 1, '0'),
        })


if __name__ == '__main__':
    unittest.main()
