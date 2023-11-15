import json
from dotenv import load_dotenv
load_dotenv()
import requests

ACTIVE_MODEL = 'llama2:7b'

def classify(text): 
    output = requests.post(
        url='http://localhost:11434/api/generate',
        data=json.dumps({
            "model": ACTIVE_MODEL,
            "stream": False,
            "system": "You are a financial analyst, determining what effect a portion of an earnings call transcript will have on the company's stock price. Be as critical and skeptical as possible, not taking what they say at face value. Respond with one of: UP, DOWN, SAME", 
            "options": {
                "temperature": 0.25,
                "num_predict": 10,
                "stop": ["\n"],
            },
            "prompt": f"""
    [INST]The company has exceeded its expectations in the past quarter[/INST]
    UP
    [INST]The company has run into a lot of issues this year.[/INST]
    DOWN
    [INST]The company will continue its steady course.[/INST]
    SAME
    [INST]{text}[/INST]
    """})
    )

    return output.json()['response']



response = classify(text="""
Thanks, James. For the second quarter, revenue of $809 million and adjusted EBITDA of $186.2 million were just a bit ahead of what we had indicated early last month. GAAP earnings per share were $2.13 and included a $19 million or $0.33 benefit from tax legislation impacts, primarily in the UK. Adjusted EPS, which excludes the tax items, amortization expense and non-operating pension income, as well as other items noted in the reconciliation at the back of our press release, was $2.11. Operationally, we continue to build upon our strong foundation and have delivered another solid quarter.

Looking at total company performance, revenue continues to trend very positively with growth sequentially of over $50 million from Q1. On a year-over-year basis, revenues grew 71% or 65% organically with strength in all three segments. Order trends also continue to be robust. We exceeded $1 billion in orders during Q2, and our backlog is at a similar amount. Our 23% adjusted EBITDA for Q2 was a substantial increase over Q1 as well as Q2 of 2020. We sequentially expanded margins meaningfully in each segment. By the way, all margin values I will discuss are on an organic basis as well, meaning, excluding any acquisitions, disposition and FX impacts.

Total company adjusted EBITDA of just over $186 million represents approximately 15% sequential growth from Q1 and a 150% growth over the prior year. We continue to grow our bottom line faster than our top line. When comparing Q2 to the prior year, we grew EBITDA over twice as much as revenues even while we are investing over $5 million in technology initiatives quarterly. As a result, our strong cash flow generation persists. Operating cash flows of nearly $113 million was a record for us for a second quarter. Our profitability expansion and increasing cash flows are the benefits from the actions we took to improve our business as the pandemic hit and the ongoing focus on integrating acquisitions and managing through the challenges we continue to face, while delivering innovative solutions to our customers.

Looking at our segments. Commercial Foodservice revenues globally were up 80% organically, and when looking just at North America, the increase was approximately 77%. The international increase was 89%. Our margins continue to expand sequentially. We produced nearly 26% adjusted EBITDA for Q2. In Residential, we saw revenue up 63%. Very high levels of demand persists for our premium appliances in outdoor cooking platforms. Here too, our margins continue to expand sequentially. We grew to well over 22% for Q2. In Food Processing, revenues increased 25% and the adjusted EBITDA margin was over 23%.

Another highlight was our operating cash flows of nearly $113 million. Discipline around cash flow is core to running the business for us. We continue to demonstrate our ability to manage costs and cash, while investing, driving innovation and providing excellent service to our customers. Our total leverage ratio is down to 2.3 times, while our covenant limit is 5.5 times. We also have over $2.1 billion of current borrowing capacity.

We will continue to execute our M&A strategy, as well as investing to improve our operations and in turn, increase profitability. An incredibly important investment we have made is the MIK, the Middleby Innovation Kitchen as well as the adjoining and stunning residential showroom in Dallas. Along with having awesome facilities, we have an outrageously talented culinary team. I've been with them on a few occasions recently, and I'm willing to admit to getting spoiled by them. Being at the MIK, which by the way is the largest IoT connected kitchen in the world, is really awe-inspiring event. Looking at our list of brands on paper is impressive, seeing them physically together is amazing and enjoying the output is even better. We are proud of the breadth of what we offer. While -- although our brands are great, I felt it was imperative to somehow come up with a Top 3 list to commemorate my days feasting at the MIK. And since I won't be able to offer tours at NAFEM this year, this can serve as a quick Middleby tour to get you all hungry as the lunch hour approaches.

One brand that does not get as much recognition in Middleby family is sometimes Globe. Their preparation solutions brought to life what may have been my favorite dish. The masterfully created watermelon gazpacho hit the spot, especially given the Dallas heat. It was incredibly flavorful, a little spicy, a hint of sweet, cool and refreshing, or crazy good as our CCO aptly described it. Moving on to the culinary course, our talented chefs will occasionally admit to a bit of a bias for hard fuel-powered cooking. Coming out of the Spain, the Josper charcoal grills are always a favorite of cooks and diners alike. A Tomahawk dry aged for 90 days in our TenderChef was absolutely delicious.

And lastly, moving onto the sandwich food group, what the Plexor can do will be market changing. And with our Middleby controls, it is so easy to use that even I could operate it. The Grilled Reuben was toasted perfectly, a crispy outside with a warm savory inside. I could go on and on about the menu breadth that the Plexor can deliver. And I'll have to come back to the cookies another time as I probably should put aside my side hustle as Chief Food Officer and get back to my typical CFO duties.

So having covered Q2 and my culinary ramblings, it is time to look forward to the rest of 2021. We've provided order and backlog data in the presentation that is available at the Investors section of our website. We also provided a full-year outlook in a release last month, which I want to further address. As I've noted before, even with a solid start to the year, we're keeping our expectations at modest levels for the near term. The reason for some caution in my tone, even with the order trends, is due to the variety of supply chain issues and cost pressures we are facing.

Many positive factors do contribute to optimistic views, such as our backlog, the innovative solutions that are addressing customer challenges, and the development activity by many of our customers. However, I don't want anyone to fail to recognize the near-term risks that are impacting our cost structure and availability of raw materials and other inputs for our products. These forces do limit our ability to generate higher top line growth for the next few quarters and will play some downward pressure on margins. We are taking pricing actions to mitigate margin impacts, but we don't expect meaningful contribution from those until early '22 given our backlog levels. I will note that we are currently anticipating low-single-digit growth in overall revenue sequentially from Q2 to Q3 with margins compressing.

Also, we call that Food Processing is a lumpy business. As we look at how contracts will be fulfilled, we will likely see a temporary revenue decline for this segment when comparing Q3 to Q2. However, order strength has continued, thus Q4 should show solid growth for FPG. As we look out to Q4, we expect total company revenues could grow at least mid-single-digits from Q3, which includes the seasonal benefits typically seen in Residential at the end of the year. Given the volatility impacting the supply side of our business, it would be prudent to anticipate Q4 margins not exceeding Q2 levels.

In closing, to highlight what the full-year 2021 should deliver, Food Processing and Residential should see revenue growth in excess of 20% over 2019 levels with margins having expanded as well. Commercial Foodservice organically will be at similar levels to 2019 in terms of revenue and margins. It's important to remember that the margins, while likely at similar to '19 levels, are after considering the inflationary impacts we are facing and the investments we have been making. This demonstrates that our management actions to address costs, integrate acquired businesses and generally improve operations are generating results. In total, consolidated company margin for 2021 should exceed 2019 levels as well.

While it is a little too early to specifically be addressing 2022 and beyond, I will reiterate our commitment to the medium-term margin targets we have set for each of the segments, which are 30% for commercial and initially 25% for the other two segments. While the near term may be a little rocky, I believe we are extremely well positioned for the longer term and that our products and solutions as well as the teams and tools we have driving our success in the market every day will continue to deliver industry-leading results.

And with that, Tina, let's please open the call to questions. Thank you very much.
""")
print(response)