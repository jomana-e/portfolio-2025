// components/StreamlitEmbed.tsx
"use client";

import React from "react";

interface StreamlitEmbedProps {
  title: string;
  description: string;
  url: string;
}

export default function StreamlitEmbed({ title, description, url }: StreamlitEmbedProps) {
  return (
    <section className="my-16">
      <h2 className="text-3xl font-bold mb-2 text-gray-900">{title}</h2>
      <p className="text-gray-600 mb-4">{description}</p>
      <div className="relative overflow-hidden rounded-2xl shadow-xl border border-gray-200 bg-white">
        <iframe
          src={url}
          title={title}
          loading="lazy"
          className="w-full h-[700px] border-0"
          allow="fullscreen"
        />
      </div>
      <div className="mt-4">
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium px-5 py-2 rounded-lg shadow-md transition"
        >
          View Full App
        </a>
      </div>
    </section>
  );
}
