#!/usr/bin/env python3
import feedparser
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import openai
from pathlib import Path
from tqdm import tqdm
import json, re, os, sys, requests

'''~~~~~~~~~~~~~CONFIGURATION~~~~~~~~~~~~~'''
# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# RSS feed sources
rss_feeds = {
    "Business of Fashion": "https://www.businessoffashion.com/arc/outboundfeeds/rss/?outputType=xml",
    "WWD": "https://wwd.com/beauty-industry-news/feed/",
    "Glossy": "https://www.glossy.co/feed/",
    "Beauty Independent": "https://www.beautyindependent.com/feed/"
}

'''~~~~~~~~~~~~~FUNCTIONS~~~~~~~~~~~~~'''
def process_rss(rss_feeds):
    """Process RSS feeds and extract articles"""
    all_articles = []
    
    for source, url in rss_feeds.items():
        try:
            print(f"Fetching: {source}")
            feed = feedparser.parse(url)
            
            if not feed.entries:
                print(f"Warning: No entries found for {source}")
                continue
                
            for entry in feed.entries:
                summary_raw = entry.get("summary", "")
                
                # Clean HTML for Beauty Independent
                if source == "Beauty Independent":
                    summary_clean = BeautifulSoup(summary_raw, "html.parser").get_text(separator=" ", strip=True)
                else:
                    summary_clean = summary_raw

                article = {
                    "source": source,
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "summary": summary_clean,
                    "tags": [tag['term'] for tag in entry.get("tags", [])] if "tags" in entry else [],
                    "published": entry.get("published", "")
                }

                # Convert published date to ISO format if available
                try:
                    if entry.get("published_parsed"):
                        article["published"] = datetime(*entry.published_parsed[:6]).isoformat()
                except Exception as e:
                    print(f"Date conversion error for {source}: {e}")
                    pass

                all_articles.append(article)
                
        except Exception as e:
            print(f"Error processing {source}: {e}")
            continue

    return pd.DataFrame(all_articles)

def gpt_filter_articles(articles, model="gpt-4"):
    """Filter articles using GPT based on user preferences"""
    filtered_articles = []
    
    # Default prompt if prompt.txt doesn't exist
    default_prompt = """
    Rate this article from 1-10 based on relevance to beauty, fashion, and business news.
    Also categorize it with PRIMARY and SECONDARY tags.
    
    Respond in this exact format:
    SCORE: [1-10]
    PRIMARY: [main category]
    SECONDARY: [sub category]
    """
    
    try:
        user_prefs = Path("prompt.txt").read_text().strip()
    except FileNotFoundError:
        print("Warning: prompt.txt not found, using default prompt")
        user_prefs = default_prompt

    for _, row in tqdm(articles.iterrows(), desc="Filtering articles"):
        try:
            prompt = f"""
            Here is article information:
            Title: {row['title']}
            Tags: {row.get('tags', '')}
            Summary: {row['summary']}
            Here is your task:
            {user_prefs}
            """

            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            answer = response.choices[0].message.content.strip().upper()
            article_data = row.to_dict()
            
            # Parse GPT response
            if "SCORE:" in answer:
                score_line = [line for line in answer.splitlines() if line.startswith("SCORE:")]
                score = int(score_line[0].replace("SCORE:", "").strip()) if score_line else 5
                article_data['score'] = score
            else:
                article_data['score'] = 5
                
            if "PRIMARY:" in answer:
                p_line = [line for line in answer.splitlines() if line.startswith("PRIMARY:")]
                prim = p_line[0].replace("PRIMARY:", "").strip() if p_line else 'General'
                article_data['Primary'] = prim
            else:
                article_data['Primary'] = 'General'
                
            if "SECONDARY:" in answer:
                s_line = [line for line in answer.splitlines() if line.startswith("SECONDARY:")]
                secon = s_line[0].replace("SECONDARY:", "").strip() if s_line else 'News'
                article_data['Secondary'] = secon
            else:
                article_data['Secondary'] = 'News'
                
            filtered_articles.append(article_data)
            
        except Exception as e:
            print(f"Error filtering article: {e}")
            # Add article with default values
            article_data = row.to_dict()
            article_data['score'] = 5
            article_data['Primary'] = 'General'
            article_data['Secondary'] = 'News'
            filtered_articles.append(article_data)

    return pd.DataFrame(filtered_articles)

def gpt_group_articles(articles_df, model="gpt-4.1"):
    """Group articles by similar events using GPT"""
    if len(articles_df) < 2:
        return []
        
    try:
        entries = []
        for i, row in articles_df.iterrows():
            entries.append({
                "index": i,
                "source": row["source"],
                "title": row["title"],
                "summary": row["summary"][:500] + "..." if len(row["summary"]) > 500 else row["summary"]
            })

        prompt = f"""
        You are analyzing business news articles to identify when multiple publications report on the EXACT SAME NEWS EVENT.

        Rules for grouping articles:
        1. Articles must describe the SAME SPECIFIC EVENT
        2. Articles must be from AT LEAST 2 DIFFERENT sources
        3. Do NOT group articles about similar but separate events
        4. Only group if core facts clearly match

        Here are the articles:
        {json.dumps(entries, indent=2)}

        Return ONLY a JSON array of groups (arrays of indices). If no valid groups exist, return [].
        Format: [[index1, index2], [index3, index4]]
        """

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\[(?:\s*\[[\d,\s]*\](?:\s*,\s*)?)*\s*\]', text)
        
        if json_match:
            text = json_match.group(0)
            groups = json.loads(text)
            
            # Validate groups
            valid_groups = []
            for group in groups:
                if len(group) >= 2:
                    sources_in_group = set(articles_df.iloc[group]['source'].tolist())
                    if len(sources_in_group) >= 2:
                        valid_groups.append(group)
            return valid_groups
        else:
            return []
            
    except Exception as e:
        print(f"Error grouping articles: {e}")
        return []

def create_grouped_dataframe(articles_df, groups):
    """Create dataframe with group information"""
    articles_df = articles_df.copy()
    articles_df['group_id'] = 'ungrouped'
    articles_df['group_size'] = 1
    
    # Assign group IDs
    for i, group in enumerate(groups):
        group_id = f'group_{i+1}'
        for article_idx in group:
            if article_idx < len(articles_df):
                articles_df.loc[articles_df.index[article_idx], 'group_id'] = group_id
                articles_df.loc[articles_df.index[article_idx], 'group_size'] = len(group)
    
    # Sort by group, then by score
    def sort_key(row):
        if row['group_id'] == 'ungrouped':
            return (999, -row['score'])
        else:
            group_num = int(row['group_id'].split('_')[1])
            return (group_num, -row['score'])
    
    articles_df['sort_key'] = articles_df.apply(sort_key, axis=1)
    articles_df = articles_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    return articles_df

def generate_newsletter_html(df, csv_url):
    """Generate HTML newsletter content"""
    html_content = f"""
<html>
<head>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background-color: #f9f9f9;
      padding: 20px;
      color: #333;
    }}
    .newsletter {{
      max-width: 700px;
      margin: auto;
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }}
    .article-group {{
      margin-bottom: 40px;
      border-bottom: 2px solid #E97451;
      padding-bottom: 20px;
    }}
    .single-article {{
      margin-bottom: 30px;
      border-bottom: 1px solid #eee;
      padding-bottom: 15px;
    }}
    .group-header {{
      font-size: 18px;
      font-weight: bold;
      color: #E97451;
      margin-bottom: 15px;
      padding: 10px;
      background-color: #fff5f3;
      border-left: 4px solid #E97451;
    }}
    .single-title {{
      font-size: 16px;
      font-weight: bold;
      color: #E97451;
      margin-bottom: 10px;
    }}
    .coverage {{
      margin-bottom: 15px;
      font-size: 14px;
      line-height: 1.6;
      background-color: #f8f9fa;
      padding: 12px;
      border-radius: 6px;
      border-left: 3px solid #dee2e6;
    }}
    .source-info {{
      font-weight: bold;
      color: #495057;
      margin-bottom: 8px;
    }}
    .summary {{
      margin: 8px 0;
      line-height: 1.5;
    }}
    .tags {{
      font-size: 12px;
      color: #6c757d;
      background-color: #e9ecef;
      padding: 5px 8px;
      border-radius: 4px;
      margin-top: 8px;
    }}
    .score {{
      font-weight: bold;
      color: #E97451;
    }}
    a {{
      color: #0073e6;
      text-decoration: none;
    }}
    .group-count {{
      font-size: 12px;
      color: #6c757d;
      font-weight: normal;
    }}
  </style>
</head>
<body>
  <div class="newsletter">
    <h2>📰 Daily Highlights – Top Stories</h2>
    <
    <p>📄 <a href="{csv_url}" download="newsletter_data.csv">Click here to download the full list as CSV</a></p>
"""
    #<p>📄 <a href="{csv_url}">Click here to view the full list as CSV</a></p>
    processed_groups = set()
    
    for _, row in df.iterrows():
        group_id = row['group_id']
        
        if group_id != 'ungrouped' and group_id not in processed_groups:
            # Handle grouped articles
            processed_groups.add(group_id)
            group_articles = df[df['group_id'] == group_id].sort_values('score', ascending=False)
            group_title = group_articles.iloc[0]['title']
            
            html_content += f"""
    <div class="article-group">
      <div class="group-header">
        {group_title}
        <span class="group-count">({len(group_articles)} sources)</span>
      </div>
"""
            
            for _, article in group_articles.iterrows():
                try:
                    published_date = pd.to_datetime(article['published']).strftime('%B %d, %Y') if article['published'] else 'Date not available'
                except:
                    published_date = 'Date not available'
                
                html_content += f"""
      <div class="coverage">
        <div class="source-info">
          {article['source']} (<a href="{article['url']}">link</a>) - {published_date}
        </div>
        <div class="summary">{article['summary']}</div>
        <div class="tags">
          <strong>Primary:</strong> {article['Primary']} | 
          <strong>Secondary:</strong> {article['Secondary']} | 
          <span class="score">Score: {article['score']}</span>
        </div>
      </div>
"""
            
            html_content += "    </div>\n"
            
        elif group_id == 'ungrouped':
            # Handle single articles
            try:
                published_date = pd.to_datetime(row['published']).strftime('%B %d, %Y') if row['published'] else 'Date not available'
            except:
                published_date = 'Date not available'
            
            html_content += f"""
    <div class="single-article">
      <div class="single-title">{row['title']}</div>
      <div class="coverage">
        <div class="source-info">
          {row['source']} (<a href="{row['url']}">link</a>) - {published_date}
        </div>
        <div class="summary">{row['summary']}</div>
        <div class="tags">
          <strong>Primary:</strong> {row['Primary']} | 
          <strong>Secondary:</strong> {row['Secondary']} | 
          <span class="score">Score: {row['score']}</span>
        </div>
      </div>
    </div>
"""

    html_content += """
  </div>
</body>
</html>
"""
    
    return html_content

def trigger_zapier_webhook(csv_url, html_content, date_str):
    """Trigger Zapier webhook to send email"""
    if not ZAPIER_WEBHOOK_URL:
        print("Warning: ZAPIER_WEBHOOK_URL not set, skipping email")
        return False
        
    try:
        payload = {
            "sender_email": "nahajtarshi@gmail.com",
            "recipient_email": "nahajtarshi@gmail.com",
            "subject": f"📰 Your Daily Newsletter",
            "html_content": html_content,
            "csv_url": csv_url,
            "date": date_str
        }
        
        response = requests.post(
            ZAPIER_WEBHOOK_URL,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Zapier webhook triggered successfully")
            return True
        else:
            print(f"❌ Zapier webhook failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error triggering Zapier webhook: {e}")
        return False

'''~~~~~~~~~~~~~MAIN EXECUTION~~~~~~~~~~~~~'''
def main():
    try:
        print("🚀 Starting newsletter generation...")
        
        # Create newsletter directory
        Path("newsletter").mkdir(exist_ok=True)
        
        # Process RSS feeds
        print("📡 Processing RSS feeds...")
        fulllist = process_rss(rss_feeds)
        
        if fulllist.empty:
            print("❌ No articles found, exiting")
            return
            
        time_str = datetime.now().strftime('%Y-%m-%d')
        
        # Save full list
        fulllist.to_csv(f"newsletter/curated_articles_{time_str}.csv", index=False)
        print(f"💾 Saved {len(fulllist)} articles to curated_articles_{time_str}.csv")

        # Filter articles
        print("🤖 Filtering articles with GPT...")
        shortlist = gpt_filter_articles(fulllist)
        shortlist = shortlist.sort_values(by='score', ascending=False)
        print(f"✅ Filtered to {len(shortlist)} articles")

        # Group articles
        print("🔗 Grouping related articles...")
        article_groups = gpt_group_articles(shortlist)
        final_df = create_grouped_dataframe(shortlist, article_groups)
        print(f"📊 Found {len(article_groups)} article groups")

        # Save grouped results
        final_csv_path = f"newsletter/shortlist_grouped_{time_str}.csv"
        final_df.to_csv(final_csv_path, index=False)
        
        # Generate CSV URL (public repo)
        csv_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{final_csv_path}"
        csv_download_url = f"https://github.com/{GITHUB_REPO}/raw/main/{final_csv_path}"
        
        # Generate newsletter HTML
        print("📝 Generating newsletter HTML...")
        newsletter_df = final_df.head(20)
        html_content = generate_newsletter_html(newsletter_df, csv_download_url)
        
        # # Save HTML
        # html_path = f"newsletter/newsletter_{time_str}.html"
        # Path(html_path).write_text(html_content, encoding='utf-8')
        # print(f"💾 Saved newsletter HTML to {html_path}")
        
        # Trigger Zapier webhook
        print("📧 Triggering Zapier webhook...")
        webhook_success = trigger_zapier_webhook(csv_download_url, html_content, time_str)
        
        # Print summary
        print("\n📈 Newsletter Summary:")
        print(f"Total articles processed: {len(fulllist)}")
        print(f"Articles in newsletter: {len(newsletter_df)}")
        print(f"Grouped articles: {len(newsletter_df[newsletter_df['group_id'] != 'ungrouped'])}")
        print(f"Single articles: {len(newsletter_df[newsletter_df['group_id'] == 'ungrouped'])}")
        print(f"CSV URL: {csv_url}")
        print(f"Zapier webhook: {'✅ Success' if webhook_success else '❌ Failed'}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()